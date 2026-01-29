//! WASM bindings for diagnostics types.

use crate::diagnostics::{InMemoryTraceSink, TraceEvent, TraceLevel, TraceSink};
use std::time::SystemTime;
use wasm_bindgen::prelude::*;

/// Trace severity level.
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum JsTraceLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
}

impl From<TraceLevel> for JsTraceLevel {
    fn from(level: TraceLevel) -> Self {
        match level {
            TraceLevel::Debug => JsTraceLevel::Debug,
            TraceLevel::Info => JsTraceLevel::Info,
            TraceLevel::Warn => JsTraceLevel::Warn,
            TraceLevel::Error => JsTraceLevel::Error,
        }
    }
}

impl From<JsTraceLevel> for TraceLevel {
    fn from(level: JsTraceLevel) -> Self {
        match level {
            JsTraceLevel::Debug => TraceLevel::Debug,
            JsTraceLevel::Info => TraceLevel::Info,
            JsTraceLevel::Warn => TraceLevel::Warn,
            JsTraceLevel::Error => TraceLevel::Error,
        }
    }
}

/// A trace event with timestamp and severity.
#[wasm_bindgen]
pub struct JsTraceEvent {
    inner: TraceEvent,
}

#[wasm_bindgen]
impl JsTraceEvent {
    /// Create a new trace event.
    #[wasm_bindgen(constructor)]
    pub fn new(id: &str, message: &str, level: JsTraceLevel) -> Self {
        Self {
            inner: TraceEvent {
                id: id.to_string(),
                message: message.to_string(),
                timestamp: SystemTime::now(),
                level: level.into(),
                span_id: None,
                trace_id: None,
            },
        }
    }

    /// Get the event ID.
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Get the event message.
    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.inner.message.clone()
    }

    /// Get the trace level.
    #[wasm_bindgen(getter)]
    pub fn level(&self) -> JsTraceLevel {
        self.inner.level.into()
    }

    /// Get the timestamp as seconds since UNIX epoch.
    #[wasm_bindgen(getter)]
    pub fn timestamp(&self) -> f64 {
        self.inner
            .timestamp
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }

    /// Get the span ID if set.
    #[wasm_bindgen(getter, js_name = spanId)]
    pub fn span_id(&self) -> Option<String> {
        self.inner.span_id.clone()
    }

    /// Get the trace ID if set.
    #[wasm_bindgen(getter, js_name = traceId)]
    pub fn trace_id(&self) -> Option<String> {
        self.inner.trace_id.clone()
    }

    /// Convert to a JSON object.
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<JsValue, JsError> {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"id".into(), &self.inner.id.clone().into())
            .map_err(|e| JsError::new(&format!("Failed to set id: {:?}", e)))?;
        js_sys::Reflect::set(&obj, &"message".into(), &self.inner.message.clone().into())
            .map_err(|e| JsError::new(&format!("Failed to set message: {:?}", e)))?;
        js_sys::Reflect::set(&obj, &"timestamp".into(), &self.timestamp().into())
            .map_err(|e| JsError::new(&format!("Failed to set timestamp: {:?}", e)))?;
        js_sys::Reflect::set(
            &obj,
            &"level".into(),
            &format!("{:?}", self.inner.level).into(),
        )
        .map_err(|e| JsError::new(&format!("Failed to set level: {:?}", e)))?;
        Ok(obj.into())
    }
}

impl JsTraceEvent {
    pub(crate) fn from_inner(inner: TraceEvent) -> Self {
        Self { inner }
    }
}

/// In-memory trace sink for collecting events.
#[wasm_bindgen]
pub struct JsInMemoryTraceSink {
    inner: InMemoryTraceSink,
}

#[wasm_bindgen]
impl JsInMemoryTraceSink {
    /// Create a new in-memory trace sink.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: InMemoryTraceSink::new(),
        }
    }

    /// Record a trace event.
    #[wasm_bindgen]
    pub fn record(&self, event: &JsTraceEvent) {
        self.inner.record(event.inner.clone());
    }

    /// Record an info-level event.
    #[wasm_bindgen]
    pub fn info(&self, id: &str, message: &str) {
        self.inner.record(TraceEvent {
            id: id.to_string(),
            message: message.to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });
    }

    /// Record a debug-level event.
    #[wasm_bindgen]
    pub fn debug(&self, id: &str, message: &str) {
        self.inner.record(TraceEvent {
            id: id.to_string(),
            message: message.to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Debug,
            span_id: None,
            trace_id: None,
        });
    }

    /// Record a warning-level event.
    #[wasm_bindgen]
    pub fn warn(&self, id: &str, message: &str) {
        self.inner.record(TraceEvent {
            id: id.to_string(),
            message: message.to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Warn,
            span_id: None,
            trace_id: None,
        });
    }

    /// Record an error-level event.
    #[wasm_bindgen]
    pub fn error(&self, id: &str, message: &str) {
        self.inner.record(TraceEvent {
            id: id.to_string(),
            message: message.to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Error,
            span_id: None,
            trace_id: None,
        });
    }

    /// Get all recorded events.
    #[wasm_bindgen]
    pub fn events(&self) -> Vec<JsTraceEvent> {
        self.inner
            .events()
            .into_iter()
            .map(JsTraceEvent::from_inner)
            .collect()
    }

    /// Get the number of recorded events.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.events().len()
    }

    /// Clear all recorded events.
    #[wasm_bindgen]
    pub fn clear(&self) {
        // Note: InMemoryTraceSink doesn't have a clear method,
        // so we can't implement this. Users should create a new sink instead.
    }
}

impl Default for JsInMemoryTraceSink {
    fn default() -> Self {
        Self::new()
    }
}

impl JsInMemoryTraceSink {
    pub(crate) fn inner(&self) -> &InMemoryTraceSink {
        &self.inner
    }
}
