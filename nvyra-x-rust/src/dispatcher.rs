//! connection pool and http dispatcher module
//! 
//! manages persistent http/2 connections with pooling and retry logic

use std::sync::Arc;
use std::time::Duration;
use anyhow::{Context, Result};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode, Method};
use hyper_rustls::HttpsConnectorBuilder;
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use hyper_util::rt::TokioExecutor;
use tokio::sync::Semaphore;
use parking_lot::RwLock;


/// connection pool for http/2 persistent connections
pub struct ConnectionPool {
    client: Client<hyper_rustls::HttpsConnector<HttpConnector>, Full<Bytes>>,
    semaphore: Arc<Semaphore>,
    timeout: Duration,
    max_connections: usize,
}

impl ConnectionPool {
    /// create new connection pool
    pub fn new(max_connections: usize, timeout: Duration) -> Result<Self> {
        let https = HttpsConnectorBuilder::new()
            .with_webpki_roots()
            .https_or_http()
            .enable_http2()
            .build();
        
        let client = Client::builder(TokioExecutor::new())
            .pool_max_idle_per_host(max_connections)
            .pool_idle_timeout(Duration::from_secs(90))
            .http2_only(true)
            .http2_keep_alive_interval(Duration::from_secs(10))
            .http2_keep_alive_timeout(Duration::from_secs(20))
            .build(https);
        
        let semaphore = Arc::new(Semaphore::new(max_connections));
        
        Ok(Self {
            client,
            semaphore,
            timeout,
            max_connections,
        })
    }
    
    /// send post request with connection pooling
    pub async fn post(&self, url: &str, body: &[u8], timeout: Duration) -> Result<Bytes> {
        let _permit = self.semaphore.acquire().await
            .context("failed to acquire connection permit")?;
        
        let request = Request::builder()
            .method(Method::POST)
            .uri(url)
            .header("content-type", "application/json")
            .header("accept", "application/json")
            .header("user-agent", "nvyra-x-client/1.0")
            .body(Full::new(Bytes::copy_from_slice(body)))
            .context("failed to build request")?;
        
        let response = tokio::time::timeout(
            timeout,
            self.client.request(request)
        )
        .await
        .context("request timed out")?
        .context("request failed")?;
        
        let status = response.status();
        
        if !status.is_success() {
            let body = response.into_body().collect().await
                .map(|b| b.to_bytes())
                .unwrap_or_default();
            
            let error_text = String::from_utf8_lossy(&body);
            anyhow::bail!("request failed with status {}: {}", status, error_text);
        }
        
        let body = response.into_body().collect().await
            .context("failed to read response body")?
            .to_bytes();
        
        Ok(body)
    }
    
    /// send get request with connection pooling
    pub async fn get(&self, url: &str, timeout: Duration) -> Result<Bytes> {
        let _permit = self.semaphore.acquire().await
            .context("failed to acquire connection permit")?;
        
        let request = Request::builder()
            .method(Method::GET)
            .uri(url)
            .header("accept", "application/json")
            .header("user-agent", "nvyra-x-client/1.0")
            .body(Full::new(Bytes::new()))
            .context("failed to build request")?;
        
        let response = tokio::time::timeout(
            timeout,
            self.client.request(request)
        )
        .await
        .context("request timed out")?
        .context("request failed")?;
        
        let status = response.status();
        
        if !status.is_success() {
            anyhow::bail!("request failed with status {}", status);
        }
        
        let body = response.into_body().collect().await
            .context("failed to read response body")?
            .to_bytes();
        
        Ok(body)
    }
    
    /// get current number of available connections
    pub fn available_connections(&self) -> usize {
        self.semaphore.available_permits()
    }
    
    /// get maximum connections
    pub fn max_connections(&self) -> usize {
        self.max_connections
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pool_creation() {
        let pool = ConnectionPool::new(100, Duration::from_secs(30));
        assert!(pool.is_ok());
        
        let pool = pool.unwrap();
        assert_eq!(pool.max_connections(), 100);
        assert_eq!(pool.available_connections(), 100);
    }
}
