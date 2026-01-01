//! nvyra-x high-performance client library
//! 
//! exposes core modules for external usage

pub mod batcher;
pub mod circuit_breaker;
pub mod dispatcher;
pub mod metrics;

pub use batcher::AdaptiveBatcher;
pub use circuit_breaker::CircuitBreaker;
pub use dispatcher::ConnectionPool;
pub use metrics::MetricsCollector;
