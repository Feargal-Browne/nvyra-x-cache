//! nvyra-x high-performance inference client
//! 
//! state-of-the-art rust implementation targeting 10,000+ requests per second
//! optimized for h200 gpu batch processing with adaptive batching and circuit breakers

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::path::PathBuf;
use std::io::BufReader;

use anyhow::{Context, Result};
use bytes::Bytes;
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use flume;
use futures::stream::{self, StreamExt};
use governor::{Quota, RateLimiter};
use hashbrown::HashMap;
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use hyper_util::rt::TokioExecutor;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use parking_lot::{Mutex, RwLock};
use prometheus::{Counter, Gauge, Histogram, HistogramOpts, IntCounter, Registry};
use serde::{Deserialize, Serialize};
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader as AsyncBufReader};
use tokio::sync::Semaphore;
use tracing::{debug, error, info, instrument, warn, Level};
use uuid::Uuid;

mod batcher;
mod dispatcher;
mod circuit_breaker;
mod metrics;

use batcher::AdaptiveBatcher;
use dispatcher::ConnectionPool;
use circuit_breaker::CircuitBreaker;
use metrics::MetricsCollector;


/// command line arguments
#[derive(Parser, Debug)]
#[command(name = "nvyra-x-client")]
#[command(about = "high-performance inference client for nvyra-x")]
#[command(version)]
struct Args {
    /// modal endpoint url for pro tier (h200)
    #[arg(long, env = "NVYRA_PRO_URL")]
    pro_url: Option<String>,
    
    /// modal endpoint url for free tier (cpu)
    #[arg(long, env = "NVYRA_FREE_URL")]
    free_url: Option<String>,
    
    /// input file path (csv, jsonl, or arrow)
    #[arg(short, long)]
    input: PathBuf,
    
    /// output file path for results
    #[arg(short, long, default_value = "results.jsonl")]
    output: PathBuf,
    
    /// target tier: pro or free
    #[arg(short, long, default_value = "pro")]
    tier: String,
    
    /// maximum concurrent connections
    #[arg(long, default_value_t = 500)]
    max_connections: usize,
    
    /// initial batch size (will adapt)
    #[arg(long, default_value_t = 128)]
    batch_size: usize,
    
    /// maximum batch size
    #[arg(long, default_value_t = 512)]
    max_batch_size: usize,
    
    /// request timeout in seconds
    #[arg(long, default_value_t = 60)]
    timeout_secs: u64,
    
    /// enable benchmark mode (skip output writing)
    #[arg(long, default_value_t = false)]
    benchmark: bool,
    
    /// prometheus metrics port
    #[arg(long, default_value_t = 9090)]
    metrics_port: u16,
    
    /// rate limit (requests per second, 0 = unlimited)
    #[arg(long, default_value_t = 0)]
    rate_limit: u32,
    
    /// number of worker threads
    #[arg(long)]
    workers: Option<usize>,
}


/// input claim item
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
struct ClaimItem {
    id: String,
    text: String,
    #[serde(default)]
    context: Option<String>,
}


/// batch payload for modal endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BatchPayload {
    items: Vec<VerificationRequest>,
}


/// verification request matching modal api
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VerificationRequest {
    claim: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
}


/// verification result from modal api
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VerificationResult {
    request_id: String,
    claim: String,
    verdict: String,
    confidence_score: f64,
    falsity_score: f64,
    fact_check_reasoning: String,
    disinfo_analysis: String,
    combined_reasoning: String,
    sources_used: Vec<String>,
    latency_ms: f64,
    tier: String,
    #[serde(default)]
    storage_status: Option<HashMap<String, bool>>,
}


/// pipeline statistics
struct PipelineStats {
    items_sent: AtomicU64,
    items_succeeded: AtomicU64,
    items_failed: AtomicU64,
    batches_sent: AtomicU64,
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
    total_latency_us: AtomicU64,
    circuit_trips: AtomicU64,
    retries: AtomicU64,
}

impl PipelineStats {
    fn new() -> Self {
        Self {
            items_sent: AtomicU64::new(0),
            items_succeeded: AtomicU64::new(0),
            items_failed: AtomicU64::new(0),
            batches_sent: AtomicU64::new(0),
            bytes_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
            circuit_trips: AtomicU64::new(0),
            retries: AtomicU64::new(0),
        }
    }
    
    fn throughput(&self, duration: Duration) -> f64 {
        let items = self.items_succeeded.load(Ordering::Relaxed) as f64;
        items / duration.as_secs_f64()
    }
    
    fn avg_latency_ms(&self) -> f64 {
        let items = self.items_succeeded.load(Ordering::Relaxed) as f64;
        let total_us = self.total_latency_us.load(Ordering::Relaxed) as f64;
        if items > 0.0 {
            total_us / items / 1000.0
        } else {
            0.0
        }
    }
}


/// main application state
struct Application {
    args: Args,
    stats: Arc<PipelineStats>,
    circuit_breaker: Arc<CircuitBreaker>,
    batcher: Arc<AdaptiveBatcher>,
    pool: Arc<ConnectionPool>,
    metrics: Arc<MetricsCollector>,
    rate_limiter: Option<Arc<RateLimiter<governor::state::NotKeyed, governor::state::InMemoryState, governor::clock::DefaultClock>>>,
}

impl Application {
    async fn new(args: Args) -> Result<Self> {
        let stats = Arc::new(PipelineStats::new());
        let circuit_breaker = Arc::new(CircuitBreaker::new(5, Duration::from_secs(10)));
        let batcher = Arc::new(AdaptiveBatcher::new(
            args.batch_size,
            args.max_batch_size,
            Duration::from_millis(100),
        ));
        
        let pool = Arc::new(ConnectionPool::new(
            args.max_connections,
            Duration::from_secs(args.timeout_secs),
        )?);
        
        let metrics = Arc::new(MetricsCollector::new()?);
        
        let rate_limiter = if args.rate_limit > 0 {
            Some(Arc::new(RateLimiter::direct(
                Quota::per_second(std::num::NonZeroU32::new(args.rate_limit).unwrap())
            )))
        } else {
            None
        };
        
        Ok(Self {
            args,
            stats,
            circuit_breaker,
            batcher,
            pool,
            metrics,
            rate_limiter,
        })
    }
    
    async fn run(&self) -> Result<()> {
        let start = Instant::now();
        
        info!("starting nvyra-x client");
        info!("  input: {:?}", self.args.input);
        info!("  tier: {}", self.args.tier);
        info!("  max connections: {}", self.args.max_connections);
        info!("  batch size: {} -> {}", self.args.batch_size, self.args.max_batch_size);
        
        // spawn metrics server
        let metrics_clone = self.metrics.clone();
        let metrics_port = self.args.metrics_port;
        tokio::spawn(async move {
            if let Err(e) = metrics_clone.serve(metrics_port).await {
                error!("metrics server error: {}", e);
            }
        });
        
        // setup progress bar
        let multi = MultiProgress::new();
        let pb = multi.add(ProgressBar::new_spinner());
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap()
        );
        
        // determine endpoint url
        let endpoint_url = match self.args.tier.as_str() {
            "pro" => self.args.pro_url.clone()
                .context("pro tier url required")?,
            "free" => self.args.free_url.clone()
                .context("free tier url required")?,
            _ => anyhow::bail!("unknown tier: {}", self.args.tier),
        };
        
        // create channels for pipeline
        let (item_tx, item_rx) = flume::bounded::<ClaimItem>(10_000);
        let (batch_tx, batch_rx) = flume::bounded::<Vec<ClaimItem>>(1_000);
        let (result_tx, result_rx) = flume::bounded::<VerificationResult>(10_000);
        
        // spawn reader task
        let input_path = self.args.input.clone();
        let item_tx_clone = item_tx.clone();
        let reader_handle = tokio::spawn(async move {
            Self::read_input(input_path, item_tx_clone).await
        });
        
        // spawn batcher task
        let batcher = self.batcher.clone();
        let item_rx_clone = item_rx.clone();
        let batch_tx_clone = batch_tx.clone();
        let batcher_handle = tokio::spawn(async move {
            Self::batch_items(batcher, item_rx_clone, batch_tx_clone).await
        });
        
        // spawn dispatcher tasks
        let num_dispatchers = self.args.workers.unwrap_or_else(num_cpus::get);
        let mut dispatcher_handles = Vec::new();
        
        for i in 0..num_dispatchers {
            let pool = self.pool.clone();
            let stats = self.stats.clone();
            let circuit = self.circuit_breaker.clone();
            let metrics = self.metrics.clone();
            let limiter = self.rate_limiter.clone();
            let batch_rx = batch_rx.clone();
            let result_tx = result_tx.clone();
            let url = endpoint_url.clone();
            let timeout = Duration::from_secs(self.args.timeout_secs);
            
            let handle = tokio::spawn(async move {
                Self::dispatch_batches(
                    i, pool, stats, circuit, metrics, limiter,
                    batch_rx, result_tx, url, timeout
                ).await
            });
            dispatcher_handles.push(handle);
        }
        
        // spawn writer task (if not benchmark mode)
        let writer_handle = if !self.args.benchmark {
            let output_path = self.args.output.clone();
            let result_rx_clone = result_rx.clone();
            Some(tokio::spawn(async move {
                Self::write_results(output_path, result_rx_clone).await
            }))
        } else {
            // just drain results
            let result_rx_clone = result_rx.clone();
            Some(tokio::spawn(async move {
                while result_rx_clone.recv_async().await.is_ok() {}
                Ok::<(), anyhow::Error>(())
            }))
        };
        
        // progress monitoring
        let stats_clone = self.stats.clone();
        let pb_clone = pb.clone();
        let monitor_handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(100)).await;
                let sent = stats_clone.items_sent.load(Ordering::Relaxed);
                let succeeded = stats_clone.items_succeeded.load(Ordering::Relaxed);
                let failed = stats_clone.items_failed.load(Ordering::Relaxed);
                let batches = stats_clone.batches_sent.load(Ordering::Relaxed);
                
                pb_clone.set_message(format!(
                    "sent: {} | ok: {} | err: {} | batches: {} | rps: {:.0}",
                    sent, succeeded, failed, batches,
                    stats_clone.throughput(Duration::from_secs(1))
                ));
            }
        });
        
        // wait for reader to finish
        drop(item_tx);
        reader_handle.await??;
        
        // wait for batcher to finish
        drop(batch_tx);
        batcher_handle.await??;
        
        // wait for dispatchers to finish
        for handle in dispatcher_handles {
            let _ = handle.await;
        }
        
        // wait for writer to finish
        drop(result_tx);
        if let Some(handle) = writer_handle {
            handle.await??;
        }
        
        monitor_handle.abort();
        
        let duration = start.elapsed();
        
        pb.finish_with_message(format!(
            "completed in {:.2}s | {} items | {:.0} rps | avg latency: {:.1}ms",
            duration.as_secs_f64(),
            self.stats.items_succeeded.load(Ordering::Relaxed),
            self.stats.throughput(duration),
            self.stats.avg_latency_ms()
        ));
        
        Ok(())
    }
    
    async fn read_input(path: PathBuf, tx: flume::Sender<ClaimItem>) -> Result<()> {
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        
        match extension {
            "csv" => Self::read_csv(path, tx).await,
            "jsonl" | "json" => Self::read_jsonl(path, tx).await,
            "arrow" | "ipc" => Self::read_arrow(path, tx).await,
            _ => anyhow::bail!("unsupported file format: {}", extension),
        }
    }
    
    async fn read_csv(path: PathBuf, tx: flume::Sender<ClaimItem>) -> Result<()> {
        let file = File::open(&path).await?;
        let reader = AsyncBufReader::new(file);
        let mut csv_reader = csv_async::AsyncReader::from_reader(reader);
        
        let mut records = csv_reader.deserialize::<ClaimItem>();
        while let Some(result) = records.next().await {
            match result {
                Ok(item) => {
                    if tx.send_async(item).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    warn!("csv parse error: {}", e);
                }
            }
        }
        
        Ok(())
    }
    
    async fn read_jsonl(path: PathBuf, tx: flume::Sender<ClaimItem>) -> Result<()> {
        let file = File::open(&path).await?;
        let reader = AsyncBufReader::new(file);
        let mut lines = reader.lines();
        
        while let Some(line) = lines.next_line().await? {
            match serde_json::from_str::<ClaimItem>(&line) {
                Ok(item) => {
                    if tx.send_async(item).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    warn!("json parse error: {}", e);
                }
            }
        }
        
        Ok(())
    }
    
    async fn read_arrow(path: PathBuf, tx: flume::Sender<ClaimItem>) -> Result<()> {
        use arrow::ipc::reader::FileReader;
        use std::fs::File as StdFile;
        
        let file = StdFile::open(&path)?;
        let reader = FileReader::try_new(file, None)?;
        
        for batch_result in reader {
            let batch = batch_result?;
            
            let id_col = batch.column_by_name("id")
                .context("missing 'id' column")?;
            let text_col = batch.column_by_name("text")
                .context("missing 'text' column")?;
            
            let id_array = id_col.as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .context("id column is not string")?;
            let text_array = text_col.as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .context("text column is not string")?;
            
            for i in 0..batch.num_rows() {
                let item = ClaimItem {
                    id: id_array.value(i).to_string(),
                    text: text_array.value(i).to_string(),
                    context: None,
                };
                
                if tx.send_async(item).await.is_err() {
                    return Ok(());
                }
            }
        }
        
        Ok(())
    }
    
    async fn batch_items(
        batcher: Arc<AdaptiveBatcher>,
        rx: flume::Receiver<ClaimItem>,
        tx: flume::Sender<Vec<ClaimItem>>,
    ) -> Result<()> {
        let mut current_batch = Vec::with_capacity(batcher.current_size());
        let mut last_flush = Instant::now();
        
        loop {
            let timeout = batcher.flush_interval();
            
            match tokio::time::timeout(timeout, rx.recv_async()).await {
                Ok(Ok(item)) => {
                    current_batch.push(item);
                    
                    if current_batch.len() >= batcher.current_size() {
                        if tx.send_async(std::mem::take(&mut current_batch)).await.is_err() {
                            break;
                        }
                        current_batch = Vec::with_capacity(batcher.current_size());
                        last_flush = Instant::now();
                    }
                }
                Ok(Err(_)) => {
                    // channel closed
                    if !current_batch.is_empty() {
                        let _ = tx.send_async(current_batch).await;
                    }
                    break;
                }
                Err(_) => {
                    // timeout - flush partial batch
                    if !current_batch.is_empty() {
                        if tx.send_async(std::mem::take(&mut current_batch)).await.is_err() {
                            break;
                        }
                        current_batch = Vec::with_capacity(batcher.current_size());
                        last_flush = Instant::now();
                    }
                }
            }
        }
        
        Ok(())
    }
    
    async fn dispatch_batches(
        worker_id: usize,
        pool: Arc<ConnectionPool>,
        stats: Arc<PipelineStats>,
        circuit: Arc<CircuitBreaker>,
        metrics: Arc<MetricsCollector>,
        limiter: Option<Arc<RateLimiter<governor::state::NotKeyed, governor::state::InMemoryState, governor::clock::DefaultClock>>>,
        rx: flume::Receiver<Vec<ClaimItem>>,
        tx: flume::Sender<VerificationResult>,
        endpoint_url: String,
        timeout: Duration,
    ) -> Result<()> {
        while let Ok(batch) = rx.recv_async().await {
            // check circuit breaker
            if !circuit.is_closed() {
                stats.circuit_trips.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(circuit.recovery_time()).await;
                continue;
            }
            
            // rate limiting
            if let Some(ref limiter) = limiter {
                limiter.until_ready().await;
            }
            
            let batch_size = batch.len();
            stats.items_sent.fetch_add(batch_size as u64, Ordering::Relaxed);
            stats.batches_sent.fetch_add(1, Ordering::Relaxed);
            
            // convert to request payload
            let requests: Vec<VerificationRequest> = batch.iter().map(|item| {
                VerificationRequest {
                    claim: item.text.clone(),
                    context: item.context.clone(),
                    request_id: Some(item.id.clone()),
                }
            }).collect();
            
            let start = Instant::now();
            
            // send request with retry
            let result = Self::send_with_retry(
                &pool,
                &endpoint_url,
                &requests,
                timeout,
                3,
            ).await;
            
            let latency = start.elapsed();
            
            match result {
                Ok(responses) => {
                    circuit.record_success();
                    metrics.record_latency(latency);
                    
                    for response in responses {
                        stats.items_succeeded.fetch_add(1, Ordering::Relaxed);
                        stats.total_latency_us.fetch_add(
                            latency.as_micros() as u64,
                            Ordering::Relaxed
                        );
                        
                        let _ = tx.send_async(response).await;
                    }
                }
                Err(e) => {
                    circuit.record_failure();
                    stats.items_failed.fetch_add(batch_size as u64, Ordering::Relaxed);
                    error!("batch dispatch failed: {}", e);
                }
            }
        }
        
        Ok(())
    }
    
    async fn send_with_retry(
        pool: &ConnectionPool,
        url: &str,
        requests: &[VerificationRequest],
        timeout: Duration,
        max_retries: usize,
    ) -> Result<Vec<VerificationResult>> {
        let payload = serde_json::to_vec(requests)?;
        
        for attempt in 0..max_retries {
            match pool.post(url, &payload, timeout).await {
                Ok(response_bytes) => {
                    let results: Vec<VerificationResult> = serde_json::from_slice(&response_bytes)?;
                    return Ok(results);
                }
                Err(e) if attempt < max_retries - 1 => {
                    let delay = Duration::from_millis(100 * (1 << attempt));
                    warn!("request failed (attempt {}), retrying in {:?}: {}", attempt + 1, delay, e);
                    tokio::time::sleep(delay).await;
                }
                Err(e) => return Err(e),
            }
        }
        
        anyhow::bail!("max retries exceeded")
    }
    
    async fn write_results(
        path: PathBuf,
        rx: flume::Receiver<VerificationResult>,
    ) -> Result<()> {
        use tokio::io::AsyncWriteExt;
        
        let file = tokio::fs::File::create(&path).await?;
        let mut writer = tokio::io::BufWriter::new(file);
        
        while let Ok(result) = rx.recv_async().await {
            let line = serde_json::to_string(&result)?;
            writer.write_all(line.as_bytes()).await?;
            writer.write_all(b"\n").await?;
        }
        
        writer.flush().await?;
        
        Ok(())
    }
}


#[tokio::main]
async fn main() -> Result<()> {
    // initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();
    
    let args = Args::parse();
    let app = Application::new(args).await?;
    app.run().await
}
