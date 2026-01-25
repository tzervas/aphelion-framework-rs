use crate::diagnostics::TraceSink;

/// Backend abstraction for model builds and execution.
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn device(&self) -> &str;
}

/// Simple backend for testing and default usage.
#[derive(Debug, Clone)]
pub struct NullBackend {
    device: String,
}

impl NullBackend {
    pub fn new(device: impl Into<String>) -> Self {
        Self {
            device: device.into(),
        }
    }

    pub fn cpu() -> Self {
        Self::new("cpu")
    }
}

impl Backend for NullBackend {
    fn name(&self) -> &str {
        "null"
    }

    fn device(&self) -> &str {
        &self.device
    }
}

/// Simple model builder interface.
pub trait ModelBuilder: Send + Sync {
    type Output;

    fn build(&self, backend: &dyn Backend, trace: &dyn TraceSink) -> Self::Output;
}
