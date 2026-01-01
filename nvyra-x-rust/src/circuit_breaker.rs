//! circuit breaker module
//! 
//! implements the circuit breaker pattern to prevent cascading failures

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use parking_lot::RwLock;

/// circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}


/// circuit breaker implementation
pub struct CircuitBreaker {
    state: RwLock<CircuitState>,
    failure_count: AtomicU32,
    success_count: AtomicU32,
    failure_threshold: u32,
    recovery_time: Duration,
    last_failure: RwLock<Option<Instant>>,
    half_open_max_calls: u32,
    half_open_calls: AtomicU32,
}

impl CircuitBreaker {
    /// create new circuit breaker
    pub fn new(failure_threshold: u32, recovery_time: Duration) -> Self {
        Self {
            state: RwLock::new(CircuitState::Closed),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            failure_threshold,
            recovery_time,
            last_failure: RwLock::new(None),
            half_open_max_calls: 3,
            half_open_calls: AtomicU32::new(0),
        }
    }
    
    /// check if circuit is closed (allowing requests)
    pub fn is_closed(&self) -> bool {
        let state = *self.state.read();
        
        match state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // check if we should transition to half-open
                if let Some(last) = *self.last_failure.read() {
                    if last.elapsed() >= self.recovery_time {
                        // transition to half-open
                        *self.state.write() = CircuitState::HalfOpen;
                        self.half_open_calls.store(0, Ordering::Relaxed);
                        tracing::info!("circuit breaker: open -> half-open");
                        return true;
                    }
                }
                false
            }
            CircuitState::HalfOpen => {
                // allow limited calls in half-open state
                let calls = self.half_open_calls.fetch_add(1, Ordering::Relaxed);
                calls < self.half_open_max_calls
            }
        }
    }
    
    /// record a successful request
    pub fn record_success(&self) {
        let mut state = self.state.write();
        
        match *state {
            CircuitState::Closed => {
                // reset failure count on success
                self.failure_count.store(0, Ordering::Relaxed);
                self.success_count.fetch_add(1, Ordering::Relaxed);
            }
            CircuitState::HalfOpen => {
                self.success_count.fetch_add(1, Ordering::Relaxed);
                
                // if we've had enough successes, close the circuit
                if self.success_count.load(Ordering::Relaxed) >= self.half_open_max_calls {
                    *state = CircuitState::Closed;
                    self.failure_count.store(0, Ordering::Relaxed);
                    self.success_count.store(0, Ordering::Relaxed);
                    tracing::info!("circuit breaker: half-open -> closed");
                }
            }
            CircuitState::Open => {
                // shouldn't happen, but handle gracefully
            }
        }
    }
    
    /// record a failed request
    pub fn record_failure(&self) {
        let mut state = self.state.write();
        
        match *state {
            CircuitState::Closed => {
                let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                
                if failures >= self.failure_threshold {
                    *state = CircuitState::Open;
                    *self.last_failure.write() = Some(Instant::now());
                    tracing::warn!(
                        "circuit breaker tripped: {} failures, entering open state",
                        failures
                    );
                }
            }
            CircuitState::HalfOpen => {
                // any failure in half-open state trips the circuit
                *state = CircuitState::Open;
                *self.last_failure.write() = Some(Instant::now());
                self.success_count.store(0, Ordering::Relaxed);
                tracing::warn!("circuit breaker: half-open -> open (failure during probe)");
            }
            CircuitState::Open => {
                // already open, just update timestamp
                *self.last_failure.write() = Some(Instant::now());
            }
        }
    }
    
    /// get recovery time duration
    pub fn recovery_time(&self) -> Duration {
        self.recovery_time
    }
    
    /// get current state
    pub fn state(&self) -> CircuitState {
        *self.state.read()
    }
    
    /// get current failure count
    pub fn failure_count(&self) -> u32 {
        self.failure_count.load(Ordering::Relaxed)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_circuit_starts_closed() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(5));
        assert!(cb.is_closed());
        assert_eq!(cb.state(), CircuitState::Closed);
    }
    
    #[test]
    fn test_circuit_opens_after_failures() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(5));
        
        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_closed());
        
        cb.record_failure();
        assert!(!cb.is_closed());
        assert_eq!(cb.state(), CircuitState::Open);
    }
    
    #[test]
    fn test_success_resets_failure_count() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(5));
        
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        
        assert_eq!(cb.failure_count(), 0);
        assert!(cb.is_closed());
    }
}
