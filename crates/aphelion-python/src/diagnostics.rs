//! Python bindings for diagnostics.

use pyo3::prelude::*;
use std::sync::Arc;
use std::time::SystemTime;

use aphelion_core::diagnostics::{InMemoryTraceSink, TraceEvent, TraceLevel, TraceSink};

/// Trace severity level.
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

/// Structured trace event.
#[pyclass(name = "TraceEvent")]
#[derive(Clone)]
pub struct PyTraceEvent {
    pub(crate) inner: TraceEvent,
}

#[pymethods]
impl PyTraceEvent {
    #[new]
    #[pyo3(signature = (id, message, level=None, span_id=None, trace_id=None))]
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

    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    #[getter]
    fn message(&self) -> &str {
        &self.inner.message
    }

    #[getter]
    fn level(&self) -> PyTraceLevel {
        self.inner.level.into()
    }

    #[getter]
    fn span_id(&self) -> Option<&str> {
        self.inner.span_id.as_deref()
    }

    #[getter]
    fn trace_id(&self) -> Option<&str> {
        self.inner.trace_id.as_deref()
    }

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

/// In-memory trace sink that collects events.
#[pyclass(name = "InMemoryTraceSink")]
#[derive(Clone)]
pub struct PyInMemoryTraceSink {
    pub(crate) inner: Arc<InMemoryTraceSink>,
}

#[pymethods]
impl PyInMemoryTraceSink {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(InMemoryTraceSink::new()),
        }
    }

    fn record(&self, event: &PyTraceEvent) {
        self.inner.record(event.inner.clone());
    }

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
