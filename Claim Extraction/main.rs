%%writefile sota_project/src/main.rs
use clap::Parser;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::PathBuf;
use std::time::Duration;
use tokio::fs::File;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    url: String, // The Modal Endpoint

    #[arg(short, long, default_value = "data.csv")]
    input: PathBuf,

    #[arg(short, long, default_value_t = 50)]
    concurrency: usize, // Number of concurrent HTTP connections
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Item {
    id: String,
    text: String,
}

#[derive(Serialize)]
struct BatchPayload {
    items: Vec<Item>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    // 1. High-Performance Client
    let client = Client::builder()
        .http2_prior_knowledge()
        .pool_max_idle_per_host(100)
        .timeout(Duration::from_secs(60)) // Allow time for H200 cold-start on first hit
        .build()?;

    println!("📂 Streaming data from: {:?}", args.input);
    let file = File::open(args.input).await?;
    let csv_stream = csv_async::AsyncReader::from_reader(file)
        .into_deserialize::<Item>();

    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner().template("{spinner:.green} {msg} {pos} batches sent")?);

    // 2. The Stream Pipeline
    csv_stream
        .map_err(|e| eprintln!("CSV Error: {}", e))
        .filter_map(|r| async { r.ok() })
        .chunks(256) // Match H200 Batch Size
        .map(|batch| {
            let client = client.clone();
            let url = args.url.clone();
            let pb = pb.clone();
            let payload = BatchPayload { items: batch };

            async move {
                let mut retries = 5;
                let mut delay = 500;

                loop {
                    let resp = client.post(&url).json(&payload).send().await;
                    match resp {
                        Ok(r) if r.status().is_success() => {
                            pb.inc(1);
                            return Ok(());
                        }
                        _ => {
                            // Backoff for wake-up logic
                            if retries == 0 { return Err("Failed"); }
                            tokio::time::sleep(Duration::from_millis(delay)).await;
                            delay *= 2;
                            retries -= 1;
                        }
                    }
                }
            }
        })
        .buffer_unordered(args.concurrency) // Parallel Request Execution
        .for_each(|_| async {})
        .await;

    pb.finish_with_message("✅ 10k Burst Ingested.");
    Ok(())
}