//! Python bindings for diagnostics.
//!
//! Structured tracing for pipeline execution. Events are typed and timestamped
//! for reliable debugging and observability.

use pyo3::prelude::*;
use std::sync::Arc;
use std::time::SystemTime;

use aphelion_core::diagnostics::{InMemoryTraceSink, TraceEvent, TraceLevel, TraceSink};

/// Severity level for trace events.
///
/// Levels follow standard logging conventions:
/// - Debug: Detailed diagnostic information
/// - Info: Normal operational events
/// - Warn: Potentially problematic situations
/// - Error: Failures that may require attention
///
/// Example:
///     >>> event = TraceEvent("stage.start", "Starting validation", TraceLevel.Info)
#[pyclass(name = "TraceLevel", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyTraceLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
}

impl From<PyTraceLevel> for TraceLevel {
    fn from(level: PyTraceLevel) -> Self {
        match level {
            PyTraceLevel::Debug => TraceLevel::Debug,
            PyTraceLevel::Info => TraceLevel::Info,
            PyTraceLevel::Warn => TraceLevel::Warn,
            PyTraceLevel::Error => TraceLevel::Error,
        }
    }
}

impl From<TraceLevel> for PyTraceLevel {
    fn from(level: TraceLevel) -> Self {
        match level {
            TraceLevel::Debug => PyTraceLevel::Debug,
            TraceLevel::Info => PyTraceLevel::Info,
            TraceLevel::Warn => PyTraceLevel::Warn,
            TraceLevel::Error => PyTraceLevel::Error,
        }
    }
}

/// Structured trace event with timestamp and metadata.
///
/// TraceEvents capture diagnostic information during pipeline execution.
/// Each event has:
/// - id: Unique identifier (e.g., "stage.validation.start")
/// - message: Human-readable description
/// - level: Severity (Debug, Info, Warn, Error)
/// - timestamp: When the event occurred
/// - Optional span_id/trace_id for distributed tracing
///
/// Example:
///     >>> event = TraceEvent("model.init", "Initializing encoder")
///     >>> event.level
///     TraceLevel.Info
///
/// Attributes:
///     id (str): Event identifier for filtering and grouping.
///     message (str): Human-readable description.
///     level (TraceLevel): Severity level.
///     span_id (str | None): Optional span ID for distributed tracing.
///     trace_id (str | None): Optional trace ID for distributed tracing.
///     timestamp_secs (float): Unix timestamp when event was created.
#[pyclass(name = "TraceEvent")]
#[derive(Clone)]
pub struct PyTraceEvent {
    pub(crate) inner: TraceEvent,
}

#[pymethods]
impl PyTraceEvent {
    /// Create a trace event.
    ///
    /// Args:
    ///     id: Event identifier (e.g., "stage.validation.start").
    ///     message: Human-readable description.
    ///     level: Severity level (default: Info).
    ///     span_id: Optional span ID for distributed tracing.
    ///     trace_id: Optional trace ID for distributed tracing.
    #[new]
    #[pyo3(signature = (id, message, level=None, span_id=None, trace_id=None))]
    #[pyo3(text_signature = "(id, message, level=None, span_id=None, trace_id=None)")]
    fn new(
        id: String,
        message: String,
        level: Option<PyTraceLevel>,
        span_id: Option<String>,
        trace_id: Option<String>,
    ) -> Self {
        Self {
            inner: TraceEvent {
                id,
                message,
                timestamp: SystemTime::now(),
                level: level.unwrap_or(PyTraceLevel::Info).into(),
                span_id,
                trace_id,
            },
        }
    }

    /// Event identifier.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// Human-readable message.
    #[getter]
    fn message(&self) -> &str {
        &self.inner.message
    }

    /// Severity level.
    #[getter]
    fn level(&self) -> PyTraceLevel {
        self.inner.level.into()
    }

    /// Optional span ID for distributed tracing.
    #[getter]
    fn span_id(&self) -> Option<&str> {
        self.inner.span_id.as_deref()
    }

    /// Optional trace ID for distributed tracing.
    #[getter]
    fn trace_id(&self) -> Option<&str> {
        self.inner.trace_id.as_deref()
    }

    /// Unix timestamp (seconds since epoch).
    #[getter]
    fn timestamp_secs(&self) -> f64 {
        self.inner
            .timestamp
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "TraceEvent(id='{}', message='{}', level={:?})",
            self.inner.id, self.inner.message, self.inner.level
        )
    }
}

/// In-memory storage for trace events.
///
/// Collects events during pipeline execution for later analysis. Thread-safe
/// and shareable across pipeline stages.
///
/// Why in-memory:
/// - Zero external dependencies (no logging framework required)
/// - Complete event capture for debugging
/// - Export to JSON for external analysis tools
///
/// For production, consider forwarding events to your observability system
/// after pipeline completion.
///
/// Example:
///     >>> trace = InMemoryTraceSink()
///     >>> trace.record(TraceEvent("test", "hello"))
///     >>> len(trace)
///     1
///     >>> for event in trace.events():
///     ...     print(event.message)
///     hello
#[pyclass(name = "InMemoryTraceSink")]
#[derive(Clone)]
pub struct PyInMemoryTraceSink {
    pub(crate) inner: Arc<InMemoryTraceSink>,
}

#[pymethods]
impl PyInMemoryTraceSink {
    /// Create an empty trace sink.
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(InMemoryTraceSink::new()),
        }
    }

    /// Record a trace event.
    ///
    /// Args:
    ///     event: TraceEvent to record.
    #[pyo3(text_signature = "(event)")]
    fn record(&self, event: &PyTraceEvent) {
        self.inner.record(event.inner.clone());
    }

    /// Get all recorded events.
    ///
    /// Returns:
    ///     List of all TraceEvents in recording order.
    fn events(&self) -> Vec<PyTraceEvent> {
        self.inner
            .events()
            .into_iter()
            .map(|e| PyTraceEvent { inner: e })
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.events().len()
    }

    fn __repr__(&self) -> String {
        format!("InMemoryTraceSink(events={})", self.inner.events().len())
    }
}

/// Enum wrapper for trace sinks.
#[derive(Clone)]
pub enum AnyTraceSink {
    InMemory(Arc<InMemoryTraceSink>),
}

impl AnyTraceSink {
    pub fn as_trace_sink(&self) -> &dyn TraceSink {
        match self {
            AnyTraceSink::InMemory(s) => s.as_ref(),
        }
    }
}

impl From<PyInMemoryTraceSink> for AnyTraceSink {
    fn from(s: PyInMemoryTraceSink) -> Self {
        AnyTraceSink::InMemory(s.inner)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTraceLevel>()?;
    m.add_class::<PyTraceEvent>()?;
    m.add_class::<PyInMemoryTraceSink>()?;
    Ok(())
}
