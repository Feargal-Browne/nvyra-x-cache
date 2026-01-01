//! adaptive batcher module
//! 
//! dynamically adjusts batch size based on latency feedback and throughput metrics

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use parking_lot::RwLock;

/// target latency for a batch in milliseconds
const TARGET_LATENCY_MS: u64 = 50;

/// minimum batch size
const MIN_BATCH_SIZE: usize = 16;

/// latency window size for averaging
const LATENCY_WINDOW_SIZE: usize = 100;


/// adaptive batcher that adjusts batch size based on latency feedback
pub struct AdaptiveBatcher {
    current_size: AtomicUsize,
    min_size: usize,
    max_size: usize,
    flush_interval: Duration,
    latency_samples: RwLock<Vec<Duration>>,
    last_adjustment: RwLock<Instant>,
    adjustment_cooldown: Duration,
}

impl AdaptiveBatcher {
    /// create new adaptive batcher
    pub fn new(initial_size: usize, max_size: usize, flush_interval: Duration) -> Self {
        Self {
            current_size: AtomicUsize::new(initial_size),
            min_size: MIN_BATCH_SIZE,
            max_size,
            flush_interval,
            latency_samples: RwLock::new(Vec::with_capacity(LATENCY_WINDOW_SIZE)),
            last_adjustment: RwLock::new(Instant::now()),
            adjustment_cooldown: Duration::from_secs(5),
        }
    }
    
    /// get current batch size
    pub fn current_size(&self) -> usize {
        self.current_size.load(Ordering::Relaxed)
    }
    
    /// get flush interval
    pub fn flush_interval(&self) -> Duration {
        self.flush_interval
    }
    
    /// record a latency sample and potentially adjust batch size
    pub fn record_latency(&self, latency: Duration) {
        let mut samples = self.latency_samples.write();
        
        if samples.len() >= LATENCY_WINDOW_SIZE {
            samples.remove(0);
        }
        samples.push(latency);
        
        // check if we should adjust
        let mut last_adj = self.last_adjustment.write();
        if last_adj.elapsed() < self.adjustment_cooldown {
            return;
        }
        
        if samples.len() < 10 {
            return;
        }
        
        // calculate average latency
        let avg_latency_ms: u64 = samples.iter()
            .map(|d| d.as_millis() as u64)
            .sum::<u64>() / samples.len() as u64;
        
        let current = self.current_size.load(Ordering::Relaxed);
        
        // adjust batch size based on latency
        let new_size = if avg_latency_ms < TARGET_LATENCY_MS / 2 {
            // latency is very low, increase batch size aggressively
            (current * 3 / 2).min(self.max_size)
        } else if avg_latency_ms < TARGET_LATENCY_MS {
            // latency is acceptable, slight increase
            (current + current / 8).min(self.max_size)
        } else if avg_latency_ms > TARGET_LATENCY_MS * 2 {
            // latency is too high, decrease batch size aggressively
            (current / 2).max(self.min_size)
        } else if avg_latency_ms > TARGET_LATENCY_MS {
            // latency is slightly high, slight decrease
            (current - current / 8).max(self.min_size)
        } else {
            current
        };
        
        if new_size != current {
            self.current_size.store(new_size, Ordering::Relaxed);
            *last_adj = Instant::now();
            tracing::info!(
                "batch size adjusted: {} -> {} (avg latency: {}ms)",
                current, new_size, avg_latency_ms
            );
        }
    }
    
    /// get statistics about the batcher
    pub fn stats(&self) -> BatcherStats {
        let samples = self.latency_samples.read();
        
        let avg_latency = if samples.is_empty() {
            Duration::ZERO
        } else {
            let total: Duration = samples.iter().sum();
            total / samples.len() as u32
        };
        
        let p99_latency = if samples.len() >= 10 {
            let mut sorted: Vec<_> = samples.iter().copied().collect();
            sorted.sort();
            sorted[sorted.len() * 99 / 100]
        } else {
            Duration::ZERO
        };
        
        BatcherStats {
            current_batch_size: self.current_size.load(Ordering::Relaxed),
            sample_count: samples.len(),
            avg_latency,
            p99_latency,
        }
    }
}


/// batcher statistics
#[derive(Debug, Clone)]
pub struct BatcherStats {
    pub current_batch_size: usize,
    pub sample_count: usize,
    pub avg_latency: Duration,
    pub p99_latency: Duration,
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batcher_initial_size() {
        let batcher = AdaptiveBatcher::new(64, 512, Duration::from_millis(100));
        assert_eq!(batcher.current_size(), 64);
    }
    
    #[test]
    fn test_batcher_increases_on_low_latency() {
        let batcher = AdaptiveBatcher::new(64, 512, Duration::from_millis(100));
        
        // simulate low latency
        for _ in 0..20 {
            batcher.record_latency(Duration::from_millis(10));
        }
        
        // wait for cooldown
        std::thread::sleep(Duration::from_secs(6));
        
        for _ in 0..20 {
            batcher.record_latency(Duration::from_millis(10));
        }
        
        assert!(batcher.current_size() >= 64);
    }
}
