//! metrics collection module
//! 
//! provides prometheus-compatible metrics for observability

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use anyhow::{Context, Result};
use prometheus::{
    Counter, Gauge, Histogram, HistogramOpts, IntCounter, Registry,
    Encoder, TextEncoder,
};
use parking_lot::RwLock;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;


/// metrics collector with prometheus export
pub struct MetricsCollector {
    registry: Registry,
    
    // counters
    requests_total: IntCounter,
    requests_success: IntCounter,
    requests_failed: IntCounter,
    bytes_sent: IntCounter,
    bytes_received: IntCounter,
    
    // gauges
    active_connections: Gauge,
    batch_size: Gauge,
    
    // histograms
    request_latency: Histogram,
    batch_latency: Histogram,
    
    // internal state
    latency_samples: RwLock<Vec<Duration>>,
}

impl MetricsCollector {
    /// create new metrics collector
    pub fn new() -> Result<Self> {
        let registry = Registry::new();
        
        let requests_total = IntCounter::new("nvyra_requests_total", "total requests sent")?;
        let requests_success = IntCounter::new("nvyra_requests_success", "successful requests")?;
        let requests_failed = IntCounter::new("nvyra_requests_failed", "failed requests")?;
        let bytes_sent = IntCounter::new("nvyra_bytes_sent", "total bytes sent")?;
        let bytes_received = IntCounter::new("nvyra_bytes_received", "total bytes received")?;
        
        let active_connections = Gauge::new("nvyra_active_connections", "currently active connections")?;
        let batch_size = Gauge::new("nvyra_batch_size", "current batch size")?;
        
        let request_latency = Histogram::with_opts(
            HistogramOpts::new("nvyra_request_latency_seconds", "request latency in seconds")
                .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
        )?;
        
        let batch_latency = Histogram::with_opts(
            HistogramOpts::new("nvyra_batch_latency_seconds", "batch processing latency in seconds")
                .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0])
        )?;
        
        // register all metrics
        registry.register(Box::new(requests_total.clone()))?;
        registry.register(Box::new(requests_success.clone()))?;
        registry.register(Box::new(requests_failed.clone()))?;
        registry.register(Box::new(bytes_sent.clone()))?;
        registry.register(Box::new(bytes_received.clone()))?;
        registry.register(Box::new(active_connections.clone()))?;
        registry.register(Box::new(batch_size.clone()))?;
        registry.register(Box::new(request_latency.clone()))?;
        registry.register(Box::new(batch_latency.clone()))?;
        
        Ok(Self {
            registry,
            requests_total,
            requests_success,
            requests_failed,
            bytes_sent,
            bytes_received,
            active_connections,
            batch_size,
            request_latency,
            batch_latency,
            latency_samples: RwLock::new(Vec::with_capacity(1000)),
        })
    }
    
    /// record a single request
    pub fn record_request(&self, success: bool, bytes_in: u64, bytes_out: u64) {
        self.requests_total.inc();
        
        if success {
            self.requests_success.inc();
        } else {
            self.requests_failed.inc();
        }
        
        self.bytes_sent.inc_by(bytes_out);
        self.bytes_received.inc_by(bytes_in);
    }
    
    /// record latency
    pub fn record_latency(&self, latency: Duration) {
        self.request_latency.observe(latency.as_secs_f64());
        
        let mut samples = self.latency_samples.write();
        if samples.len() >= 1000 {
            samples.remove(0);
        }
        samples.push(latency);
    }
    
    /// record batch latency
    pub fn record_batch_latency(&self, latency: Duration) {
        self.batch_latency.observe(latency.as_secs_f64());
    }
    
    /// set active connections gauge
    pub fn set_active_connections(&self, count: usize) {
        self.active_connections.set(count as f64);
    }
    
    /// set batch size gauge
    pub fn set_batch_size(&self, size: usize) {
        self.batch_size.set(size as f64);
    }
    
    /// get percentile latency
    pub fn percentile_latency(&self, percentile: f64) -> Duration {
        let samples = self.latency_samples.read();
        
        if samples.is_empty() {
            return Duration::ZERO;
        }
        
        let mut sorted: Vec<_> = samples.iter().copied().collect();
        sorted.sort();
        
        let idx = ((sorted.len() as f64) * percentile / 100.0) as usize;
        let idx = idx.min(sorted.len() - 1);
        
        sorted[idx]
    }
    
    /// get average latency
    pub fn avg_latency(&self) -> Duration {
        let samples = self.latency_samples.read();
        
        if samples.is_empty() {
            return Duration::ZERO;
        }
        
        let total: Duration = samples.iter().sum();
        total / samples.len() as u32
    }
    
    /// serve prometheus metrics endpoint
    pub async fn serve(&self, port: u16) -> Result<()> {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", port)).await
            .context("failed to bind metrics port")?;
        
        tracing::info!("metrics server listening on port {}", port);
        
        loop {
            let (mut socket, _) = listener.accept().await?;
            
            let encoder = TextEncoder::new();
            let metric_families = self.registry.gather();
            let mut buffer = Vec::new();
            encoder.encode(&metric_families, &mut buffer)?;
            
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\n\r\n",
                buffer.len()
            );
            
            socket.write_all(response.as_bytes()).await?;
            socket.write_all(&buffer).await?;
        }
    }
    
    /// get current stats as json
    pub fn stats_json(&self) -> serde_json::Value {
        serde_json::json!({
            "requests": {
                "total": self.requests_total.get(),
                "success": self.requests_success.get(),
                "failed": self.requests_failed.get(),
            },
            "bytes": {
                "sent": self.bytes_sent.get(),
                "received": self.bytes_received.get(),
            },
            "latency": {
                "avg_ms": self.avg_latency().as_secs_f64() * 1000.0,
                "p50_ms": self.percentile_latency(50.0).as_secs_f64() * 1000.0,
                "p95_ms": self.percentile_latency(95.0).as_secs_f64() * 1000.0,
                "p99_ms": self.percentile_latency(99.0).as_secs_f64() * 1000.0,
            }
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_creation() {
        let metrics = MetricsCollector::new();
        assert!(metrics.is_ok());
    }
    
    #[test]
    fn test_latency_recording() {
        let metrics = MetricsCollector::new().unwrap();
        
        metrics.record_latency(Duration::from_millis(10));
        metrics.record_latency(Duration::from_millis(20));
        metrics.record_latency(Duration::from_millis(30));
        
        let avg = metrics.avg_latency();
        assert!(avg.as_millis() >= 15 && avg.as_millis() <= 25);
    }
}
