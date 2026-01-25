use std::sync::{Arc, Mutex};
use std::time::SystemTime;

#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub id: String,
    pub message: String,
    pub timestamp: SystemTime,
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
            Err(poisoned) => poisoned.into_inner(),
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
