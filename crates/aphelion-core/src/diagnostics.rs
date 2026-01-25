//! Diagnostic and tracing infrastructure for model building.
//!
//! This module provides event recording and filtering capabilities for tracking
//! model building progress, debugging issues, and collecting performance metrics.
//! Events are recorded through trait objects, enabling pluggable trace sinks.

use serde::Serialize;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Severity level for trace events.
///
/// `TraceLevel` defines the severity of diagnostic events, allowing filtering
/// and selective reporting based on importance.
///
/// # Ordering
///
/// Levels are ordered from least to most severe: Debug < Info < Warn < Error
///
/// # Examples
///
/// ```
/// use aphelion_core::diagnostics::TraceLevel;
///
/// assert!(TraceLevel::Debug < TraceLevel::Info);
/// assert!(TraceLevel::Info < TraceLevel::Warn);
/// assert!(TraceLevel::Warn < TraceLevel::Error);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum TraceLevel {
    /// Debug-level event (verbose, typically disabled in production)
    Debug,
    /// Informational event (normal operation)
    Info,
    /// Warning event (unexpected but recoverable)
    Warn,
    /// Error event (failure or critical issue)
    Error,
}

/// A diagnostic event recorded during model building.
///
/// `TraceEvent` captures information about operations happening during model construction,
/// including timing, severity level, and distributed tracing identifiers.
///
/// # Fields
///
/// * `id` - Unique event identifier
/// * `message` - Event description
/// * `timestamp` - When the event occurred
/// * `level` - Severity level
/// * `span_id` - Optional distributed tracing span identifier
/// * `trace_id` - Optional distributed tracing trace identifier
///
/// # Examples
///
/// ```
/// use aphelion_core::diagnostics::{TraceEvent, TraceLevel};
/// use std::time::SystemTime;
///
/// let event = TraceEvent {
///     id: "build.start".to_string(),
///     message: "starting model build".to_string(),
///     timestamp: SystemTime::now(),
///     level: TraceLevel::Info,
///     span_id: Some("span-123".to_string()),
///     trace_id: None,
/// };
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct TraceEvent {
    /// Unique event identifier
    pub id: String,
    /// Human-readable event description
    pub message: String,
    /// Timestamp when the event occurred
    pub timestamp: SystemTime,
    /// Severity level of the event
    pub level: TraceLevel,
    /// Optional distributed tracing span identifier
    pub span_id: Option<String>,
    /// Optional distributed tracing trace identifier
    pub trace_id: Option<String>,
}

/// Trait for recording trace events.
///
/// `TraceSink` defines the interface for consuming diagnostic events. Implementations
/// handle event storage, filtering, export, or real-time monitoring.
///
/// # Implementing TraceSink
///
/// Types implementing `TraceSink` should be thread-safe (`Send + Sync`) and handle
/// concurrent event recording from multiple threads.
///
/// # Examples
///
/// ```
/// use aphelion_core::diagnostics::{TraceSink, TraceEvent, TraceLevel};
/// use std::time::SystemTime;
/// use std::sync::{Arc, Mutex};
///
/// struct CountingSink {
///     count: Arc<Mutex<usize>>,
/// }
///
/// impl TraceSink for CountingSink {
///     fn record(&self, _event: TraceEvent) {
///         *self.count.lock().unwrap() += 1;
///     }
/// }
/// ```
pub trait TraceSink: Send + Sync {
    /// Records a trace event.
    ///
    /// # Arguments
    ///
    /// * `event` - The trace event to record
    fn record(&self, event: TraceEvent);
}

/// Extension trait providing convenience methods for recording trace events.
///
/// `TraceSinkExt` reduces boilerplate when recording diagnostic events by automatically
/// setting timestamp, and defaulting optional fields (span_id, trace_id). It provides
/// convenience methods for each severity level: `debug()`, `info()`, `warn()`, and `error()`.
///
/// # Examples
///
/// ```
/// use aphelion_core::diagnostics::{TraceSink, TraceSinkExt, InMemoryTraceSink};
///
/// let sink = InMemoryTraceSink::new();
///
/// // Simple one-liner recording
/// sink.info("stage.start", "Starting optimization stage");
/// sink.warn("config.deprecated", "Field 'legacy_mode' is deprecated");
/// sink.error("build.failed", "Failed to compile graph");
///
/// let events = sink.events();
/// assert_eq!(events.len(), 3);
/// assert_eq!(events[0].id, "stage.start");
/// assert_eq!(events[1].id, "config.deprecated");
/// assert_eq!(events[2].id, "build.failed");
/// ```
pub trait TraceSinkExt: TraceSink {
    /// Records a debug-level event with auto-generated timestamp.
    ///
    /// # Arguments
    ///
    /// * `id` - Event identifier (will be converted to String)
    /// * `message` - Event description (will be converted to String)
    fn debug(&self, id: &str, message: impl Into<String>) {
        self.record(TraceEvent {
            id: id.to_string(),
            message: message.into(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Debug,
            span_id: None,
            trace_id: None,
        });
    }

    /// Records an info-level event with auto-generated timestamp.
    ///
    /// # Arguments
    ///
    /// * `id` - Event identifier (will be converted to String)
    /// * `message` - Event description (will be converted to String)
    fn info(&self, id: &str, message: impl Into<String>) {
        self.record(TraceEvent {
            id: id.to_string(),
            message: message.into(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });
    }

    /// Records a warn-level event with auto-generated timestamp.
    ///
    /// # Arguments
    ///
    /// * `id` - Event identifier (will be converted to String)
    /// * `message` - Event description (will be converted to String)
    fn warn(&self, id: &str, message: impl Into<String>) {
        self.record(TraceEvent {
            id: id.to_string(),
            message: message.into(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Warn,
            span_id: None,
            trace_id: None,
        });
    }

    /// Records an error-level event with auto-generated timestamp.
    ///
    /// # Arguments
    ///
    /// * `id` - Event identifier (will be converted to String)
    /// * `message` - Event description (will be converted to String)
    fn error(&self, id: &str, message: impl Into<String>) {
        self.record(TraceEvent {
            id: id.to_string(),
            message: message.into(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Error,
            span_id: None,
            trace_id: None,
        });
    }
}

/// Blanket implementation of TraceSinkExt for all types implementing TraceSink.
impl<T: TraceSink + ?Sized> TraceSinkExt for T {}

/// An in-memory trace sink that stores all recorded events.
///
/// `InMemoryTraceSink` is useful for testing, debugging, and analysis of build processes.
/// All events are stored in memory and can be retrieved as a vector.
///
/// # Examples
///
/// ```
/// use aphelion_core::diagnostics::{InMemoryTraceSink, TraceSink, TraceEvent, TraceLevel};
/// use std::time::SystemTime;
///
/// let sink = InMemoryTraceSink::new();
/// let event = TraceEvent {
///     id: "test".to_string(),
///     message: "test event".to_string(),
///     timestamp: SystemTime::now(),
///     level: TraceLevel::Info,
///     span_id: None,
///     trace_id: None,
/// };
///
/// sink.record(event);
/// let events = sink.events();
/// assert_eq!(events.len(), 1);
/// ```
#[derive(Default, Clone)]
pub struct InMemoryTraceSink {
    events: Arc<Mutex<Vec<TraceEvent>>>,
}

impl InMemoryTraceSink {
    /// Creates a new empty in-memory trace sink.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::diagnostics::InMemoryTraceSink;
    ///
    /// let sink = InMemoryTraceSink::new();
    /// assert_eq!(sink.events().len(), 0);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Retrieves a copy of all recorded events.
    ///
    /// # Returns
    ///
    /// A vector containing clones of all recorded events
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::diagnostics::{InMemoryTraceSink, TraceSink, TraceEvent, TraceLevel};
    /// use std::time::SystemTime;
    ///
    /// let sink = InMemoryTraceSink::new();
    /// let event = TraceEvent {
    ///     id: "event1".to_string(),
    ///     message: "first event".to_string(),
    ///     timestamp: SystemTime::now(),
    ///     level: TraceLevel::Debug,
    ///     span_id: None,
    ///     trace_id: None,
    /// };
    /// sink.record(event);
    ///
    /// let events = sink.events();
    /// assert_eq!(events.len(), 1);
    /// assert_eq!(events[0].id, "event1");
    /// ```
    pub fn events(&self) -> Vec<TraceEvent> {
        match self.events.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }
}

impl TraceSink for InMemoryTraceSink {
    fn record(&self, event: TraceEvent) {
        match self.events.lock() {
            Ok(mut guard) => guard.push(event),
            Err(poisoned) => poisoned.into_inner().push(event),
        }
    }
}

/// Filters trace events by minimum severity level.
///
/// `TraceFilter` wraps another trace sink and only records events that meet
/// the minimum severity threshold. This enables efficient filtering without
/// creating filtered copies of events.
///
/// # Examples
///
/// ```
/// use aphelion_core::diagnostics::{TraceFilter, InMemoryTraceSink, TraceSink, TraceEvent, TraceLevel};
/// use std::sync::Arc;
/// use std::time::SystemTime;
///
/// let inner_sink = Arc::new(InMemoryTraceSink::new());
/// let filter = TraceFilter::new(TraceLevel::Warn, inner_sink.clone());
///
/// // Debug event will be filtered out
/// filter.record(TraceEvent {
///     id: "debug".to_string(),
///     message: "debug info".to_string(),
///     timestamp: SystemTime::now(),
///     level: TraceLevel::Debug,
///     span_id: None,
///     trace_id: None,
/// });
///
/// // Error event will pass through
/// filter.record(TraceEvent {
///     id: "error".to_string(),
///     message: "error info".to_string(),
///     timestamp: SystemTime::now(),
///     level: TraceLevel::Error,
///     span_id: None,
///     trace_id: None,
/// });
///
/// assert_eq!(inner_sink.events().len(), 1);
/// ```
#[derive(Clone)]
pub struct TraceFilter {
    min_level: TraceLevel,
    sink: Arc<dyn TraceSink>,
}

impl TraceFilter {
    /// Creates a new trace filter with the given minimum level.
    ///
    /// # Arguments
    ///
    /// * `min_level` - Only events with severity >= this level will be recorded
    /// * `sink` - The underlying sink to forward filtered events to
    pub fn new(min_level: TraceLevel, sink: Arc<dyn TraceSink>) -> Self {
        Self { min_level, sink }
    }
}

impl TraceSink for TraceFilter {
    fn record(&self, event: TraceEvent) {
        if event.level >= self.min_level {
            self.sink.record(event);
        }
    }
}

/// Forwards trace events to multiple sinks simultaneously.
///
/// `MultiSink` distributes each recorded event to all registered sinks,
/// enabling simultaneous logging to multiple outputs (e.g., file, memory, remote).
///
/// # Examples
///
/// ```
/// use aphelion_core::diagnostics::{MultiSink, InMemoryTraceSink, TraceSink, TraceEvent, TraceLevel};
/// use std::sync::Arc;
/// use std::time::SystemTime;
///
/// let sink1 = Arc::new(InMemoryTraceSink::new());
/// let sink2 = Arc::new(InMemoryTraceSink::new());
/// let multi = MultiSink::new(vec![sink1.clone(), sink2.clone()]);
///
/// let event = TraceEvent {
///     id: "event1".to_string(),
///     message: "test".to_string(),
///     timestamp: SystemTime::now(),
///     level: TraceLevel::Info,
///     span_id: None,
///     trace_id: None,
/// };
///
/// multi.record(event);
/// assert_eq!(sink1.events().len(), 1);
/// assert_eq!(sink2.events().len(), 1);
/// ```
#[derive(Clone)]
pub struct MultiSink {
    sinks: Arc<Vec<Arc<dyn TraceSink>>>,
}

impl MultiSink {
    /// Creates a new multi-sink from a vector of trace sinks.
    ///
    /// # Arguments
    ///
    /// * `sinks` - Vector of trace sinks to forward events to
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::diagnostics::{MultiSink, InMemoryTraceSink};
    /// use std::sync::Arc;
    ///
    /// let sink1 = Arc::new(InMemoryTraceSink::new());
    /// let sink2 = Arc::new(InMemoryTraceSink::new());
    /// let multi = MultiSink::new(vec![sink1, sink2]);
    /// ```
    pub fn new(sinks: Vec<Arc<dyn TraceSink>>) -> Self {
        Self {
            sinks: Arc::new(sinks),
        }
    }
}

impl TraceSink for MultiSink {
    fn record(&self, event: TraceEvent) {
        for sink in self.sinks.iter() {
            sink.record(event.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_level_ordering() {
        assert!(TraceLevel::Debug < TraceLevel::Info);
        assert!(TraceLevel::Info < TraceLevel::Warn);
        assert!(TraceLevel::Warn < TraceLevel::Error);
    }

    #[test]
    fn test_trace_level_equality() {
        assert_eq!(TraceLevel::Info, TraceLevel::Info);
        assert_ne!(TraceLevel::Debug, TraceLevel::Error);
    }

    #[test]
    fn test_trace_event_creation() {
        let event = TraceEvent {
            id: "test-1".to_string(),
            message: "Test message".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: Some("span-123".to_string()),
            trace_id: Some("trace-456".to_string()),
        };

        assert_eq!(event.id, "test-1");
        assert_eq!(event.message, "Test message");
        assert_eq!(event.level, TraceLevel::Info);
        assert_eq!(event.span_id, Some("span-123".to_string()));
        assert_eq!(event.trace_id, Some("trace-456".to_string()));
    }

    #[test]
    fn test_in_memory_trace_sink_record() {
        let sink = InMemoryTraceSink::new();
        let event = TraceEvent {
            id: "event-1".to_string(),
            message: "First event".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Debug,
            span_id: None,
            trace_id: None,
        };

        sink.record(event.clone());
        let events = sink.events();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, "event-1");
        assert_eq!(events[0].message, "First event");
    }

    #[test]
    fn test_in_memory_trace_sink_multiple_records() {
        let sink = InMemoryTraceSink::new();

        for i in 0..5 {
            let event = TraceEvent {
                id: format!("event-{}", i),
                message: format!("Event {}", i),
                timestamp: SystemTime::now(),
                level: TraceLevel::Info,
                span_id: None,
                trace_id: None,
            };
            sink.record(event);
        }

        let events = sink.events();
        assert_eq!(events.len(), 5);
    }

    #[test]
    fn test_trace_filter_allows_higher_levels() {
        let in_memory = Arc::new(InMemoryTraceSink::new());
        let filter = TraceFilter::new(TraceLevel::Warn, in_memory.clone());

        let debug_event = TraceEvent {
            id: "debug".to_string(),
            message: "Debug message".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Debug,
            span_id: None,
            trace_id: None,
        };

        let warn_event = TraceEvent {
            id: "warn".to_string(),
            message: "Warn message".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Warn,
            span_id: None,
            trace_id: None,
        };

        filter.record(debug_event);
        filter.record(warn_event);

        let recorded = in_memory.events();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].id, "warn");
    }

    #[test]
    fn test_trace_filter_blocks_lower_levels() {
        let in_memory = Arc::new(InMemoryTraceSink::new());
        let filter = TraceFilter::new(TraceLevel::Error, in_memory.clone());

        let warn_event = TraceEvent {
            id: "warn".to_string(),
            message: "Warn message".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Warn,
            span_id: None,
            trace_id: None,
        };

        let error_event = TraceEvent {
            id: "error".to_string(),
            message: "Error message".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Error,
            span_id: None,
            trace_id: None,
        };

        filter.record(warn_event);
        filter.record(error_event);

        let recorded = in_memory.events();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].id, "error");
    }

    #[test]
    fn test_multi_sink_forwards_to_all_sinks() {
        let sink1 = Arc::new(InMemoryTraceSink::new());
        let sink2 = Arc::new(InMemoryTraceSink::new());
        let sink3 = Arc::new(InMemoryTraceSink::new());

        let multi = MultiSink::new(vec![sink1.clone(), sink2.clone(), sink3.clone()]);

        let event = TraceEvent {
            id: "multi-event".to_string(),
            message: "Multi sink test".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        };

        multi.record(event);

        assert_eq!(sink1.events().len(), 1);
        assert_eq!(sink2.events().len(), 1);
        assert_eq!(sink3.events().len(), 1);

        assert_eq!(sink1.events()[0].id, "multi-event");
        assert_eq!(sink2.events()[0].id, "multi-event");
        assert_eq!(sink3.events()[0].id, "multi-event");
    }

    #[test]
    fn test_multi_sink_empty() {
        let multi = MultiSink::new(vec![]);

        let event = TraceEvent {
            id: "test".to_string(),
            message: "Test".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        };

        multi.record(event);
    }

    #[test]
    fn test_multi_sink_with_filter() {
        let in_memory = Arc::new(InMemoryTraceSink::new());
        let filter = Arc::new(TraceFilter::new(TraceLevel::Warn, in_memory.clone()));

        let in_memory2 = Arc::new(InMemoryTraceSink::new());

        let multi = MultiSink::new(vec![filter, in_memory2.clone()]);

        let debug_event = TraceEvent {
            id: "debug".to_string(),
            message: "Debug".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Debug,
            span_id: None,
            trace_id: None,
        };

        let warn_event = TraceEvent {
            id: "warn".to_string(),
            message: "Warn".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Warn,
            span_id: None,
            trace_id: None,
        };

        multi.record(debug_event);
        multi.record(warn_event);

        assert_eq!(in_memory.events().len(), 1);
        assert_eq!(in_memory.events()[0].id, "warn");

        assert_eq!(in_memory2.events().len(), 2);
    }

    #[test]
    fn test_trace_sink_ext_info_helper() {
        let sink = InMemoryTraceSink::new();

        // One-liner usage without boilerplate
        sink.info("stage.start", "Starting build");

        let events = sink.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, "stage.start");
        assert_eq!(events[0].message, "Starting build");
        assert_eq!(events[0].level, TraceLevel::Info);
        assert_eq!(events[0].span_id, None);
        assert_eq!(events[0].trace_id, None);
    }

    #[test]
    fn test_trace_sink_ext_debug_helper() {
        let sink = InMemoryTraceSink::new();

        sink.debug("optimizer.verbose", "Iteration 42");

        let events = sink.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, "optimizer.verbose");
        assert_eq!(events[0].message, "Iteration 42");
        assert_eq!(events[0].level, TraceLevel::Debug);
    }

    #[test]
    fn test_trace_sink_ext_warn_helper() {
        let sink = InMemoryTraceSink::new();

        sink.warn("config.deprecated", "Using legacy mode");

        let events = sink.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, "config.deprecated");
        assert_eq!(events[0].message, "Using legacy mode");
        assert_eq!(events[0].level, TraceLevel::Warn);
    }

    #[test]
    fn test_trace_sink_ext_error_helper() {
        let sink = InMemoryTraceSink::new();

        sink.error("build.failed", "Graph compilation error");

        let events = sink.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].id, "build.failed");
        assert_eq!(events[0].message, "Graph compilation error");
        assert_eq!(events[0].level, TraceLevel::Error);
    }

    #[test]
    fn test_trace_sink_ext_multiple_helpers() {
        let sink = InMemoryTraceSink::new();

        sink.debug("app.start", "Debug message");
        sink.info("app.stage1", "Info message");
        sink.warn("app.stage2", "Warn message");
        sink.error("app.stage3", "Error message");

        let events = sink.events();
        assert_eq!(events.len(), 4);
        assert_eq!(events[0].level, TraceLevel::Debug);
        assert_eq!(events[1].level, TraceLevel::Info);
        assert_eq!(events[2].level, TraceLevel::Warn);
        assert_eq!(events[3].level, TraceLevel::Error);
    }

    #[test]
    fn test_trace_sink_ext_with_string_conversion() {
        let sink = InMemoryTraceSink::new();

        // Test with String conversion via Into<String>
        let field_name = "deprecated_field";
        sink.warn(
            "config.field",
            format!("Field '{}' is deprecated", field_name),
        );

        let events = sink.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].message, "Field 'deprecated_field' is deprecated");
    }

    #[test]
    fn test_trace_sink_ext_with_filter() {
        let in_memory = Arc::new(InMemoryTraceSink::new());
        let filter = Arc::new(TraceFilter::new(TraceLevel::Warn, in_memory.clone()));

        // Debug and info should be filtered out
        filter.debug("test.debug", "Debug message");
        filter.info("test.info", "Info message");

        // Warn and error should pass through
        filter.warn("test.warn", "Warn message");
        filter.error("test.error", "Error message");

        let recorded = in_memory.events();
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].level, TraceLevel::Warn);
        assert_eq!(recorded[1].level, TraceLevel::Error);
    }

    #[test]
    fn test_trace_sink_ext_with_multi_sink() {
        let sink1 = Arc::new(InMemoryTraceSink::new());
        let sink2 = Arc::new(InMemoryTraceSink::new());
        let multi = MultiSink::new(vec![sink1.clone(), sink2.clone()]);

        multi.info("event", "Test event");

        assert_eq!(sink1.events().len(), 1);
        assert_eq!(sink2.events().len(), 1);
        assert_eq!(sink1.events()[0].message, "Test event");
        assert_eq!(sink2.events()[0].message, "Test event");
    }
}
