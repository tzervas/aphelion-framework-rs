use crate::diagnostics::{TraceEvent, TraceSink};
use serde::Serialize;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

/// Serializable representation of a TraceEvent
#[derive(Debug, Clone, Serialize)]
pub struct SerializableTraceEvent {
    pub id: String,
    pub message: String,
    pub timestamp: String,
    pub level: String,
    pub span_id: Option<String>,
    pub trace_id: Option<String>,
}

impl From<&TraceEvent> for SerializableTraceEvent {
    fn from(event: &TraceEvent) -> Self {
        let timestamp = event
            .timestamp
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let level = match event.level {
            crate::diagnostics::TraceLevel::Debug => "DEBUG",
            crate::diagnostics::TraceLevel::Info => "INFO",
            crate::diagnostics::TraceLevel::Warn => "WARN",
            crate::diagnostics::TraceLevel::Error => "ERROR",
        };

        Self {
            id: event.id.clone(),
            message: event.message.clone(),
            timestamp,
            level: level.to_string(),
            span_id: event.span_id.clone(),
            trace_id: event.trace_id.clone(),
        }
    }
}

/// JSON exporter for trace events
pub struct JsonExporter {
    events: Arc<Mutex<Vec<TraceEvent>>>,
}

impl JsonExporter {
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Convert all recorded events to JSON string
    pub fn to_json(&self) -> String {
        match self.events.lock() {
            Ok(guard) => {
                let serializable: Vec<SerializableTraceEvent> =
                    guard.iter().map(|e| SerializableTraceEvent::from(e)).collect();
                serde_json::to_string_pretty(&serializable).unwrap_or_else(|_| "[]".to_string())
            }
            Err(poisoned) => {
                let guard = poisoned.into_inner();
                let serializable: Vec<SerializableTraceEvent> =
                    guard.iter().map(|e| SerializableTraceEvent::from(e)).collect();
                serde_json::to_string_pretty(&serializable).unwrap_or_else(|_| "[]".to_string())
            }
        }
    }

    /// Write all recorded events as JSON to a writer
    pub fn write_to<W: Write>(&self, mut writer: W) -> io::Result<()> {
        let json = self.to_json();
        writer.write_all(json.as_bytes())?;
        Ok(())
    }
}

impl Default for JsonExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceSink for JsonExporter {
    fn record(&self, event: TraceEvent) {
        match self.events.lock() {
            Ok(mut guard) => guard.push(event),
            Err(poisoned) => poisoned.into_inner().push(event),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::TraceLevel;
    use std::time::SystemTime;

    #[test]
    fn test_json_exporter_creation() {
        let exporter = JsonExporter::new();
        assert_eq!(exporter.to_json(), "[]");
    }

    #[test]
    fn test_json_exporter_record() {
        let exporter = JsonExporter::new();
        let event = TraceEvent {
            id: "test-1".to_string(),
            message: "Test message".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: Some("span-123".to_string()),
            trace_id: Some("trace-456".to_string()),
        };

        exporter.record(event);
        let json = exporter.to_json();
        assert!(json.contains("test-1"));
        assert!(json.contains("Test message"));
        assert!(json.contains("INFO"));
        assert!(json.contains("span-123"));
        assert!(json.contains("trace-456"));
    }

    #[test]
    fn test_json_exporter_multiple_events() {
        let exporter = JsonExporter::new();
        for i in 0..3 {
            let event = TraceEvent {
                id: format!("test-{}", i),
                message: format!("Message {}", i),
                timestamp: SystemTime::now(),
                level: match i {
                    0 => TraceLevel::Debug,
                    1 => TraceLevel::Info,
                    _ => TraceLevel::Error,
                },
                span_id: None,
                trace_id: None,
            };
            exporter.record(event);
        }

        let json = exporter.to_json();
        assert!(json.contains("test-0"));
        assert!(json.contains("test-1"));
        assert!(json.contains("test-2"));
        assert!(json.contains("DEBUG"));
        assert!(json.contains("ERROR"));
    }

    #[test]
    fn test_json_exporter_write_to() {
        let exporter = JsonExporter::new();
        let event = TraceEvent {
            id: "test-write".to_string(),
            message: "Write test".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Warn,
            span_id: None,
            trace_id: None,
        };
        exporter.record(event);

        let mut buffer = Vec::new();
        assert!(exporter.write_to(&mut buffer).is_ok());
        let output = String::from_utf8(buffer).unwrap();
        assert!(output.contains("test-write"));
        assert!(output.contains("WARN"));
    }

    #[test]
    fn test_serializable_trace_event_conversion() {
        let event = TraceEvent {
            id: "convert-test".to_string(),
            message: "Conversion test".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Error,
            span_id: Some("span-789".to_string()),
            trace_id: Some("trace-999".to_string()),
        };

        let serializable = SerializableTraceEvent::from(&event);
        assert_eq!(serializable.id, "convert-test");
        assert_eq!(serializable.message, "Conversion test");
        assert_eq!(serializable.level, "ERROR");
        assert_eq!(serializable.span_id, Some("span-789".to_string()));
        assert_eq!(serializable.trace_id, Some("trace-999".to_string()));
    }
}
