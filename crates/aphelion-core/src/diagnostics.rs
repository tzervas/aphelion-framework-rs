use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Trace level for diagnostic events
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TraceLevel {
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub id: String,
    pub message: String,
    pub timestamp: SystemTime,
    pub level: TraceLevel,
    pub span_id: Option<String>,
    pub trace_id: Option<String>,
}

pub trait TraceSink: Send + Sync {
    fn record(&self, event: TraceEvent);
}

#[derive(Default, Clone)]
pub struct InMemoryTraceSink {
    events: Arc<Mutex<Vec<TraceEvent>>>,
}

impl InMemoryTraceSink {
    pub fn new() -> Self {
        Self::default()
    }

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

/// Filters trace events by minimum level
#[derive(Clone)]
pub struct TraceFilter {
    min_level: TraceLevel,
    sink: Arc<dyn TraceSink>,
}

impl TraceFilter {
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

/// Forwards trace events to multiple sinks
#[derive(Clone)]
pub struct MultiSink {
    sinks: Arc<Vec<Arc<dyn TraceSink>>>,
}

impl MultiSink {
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
}
