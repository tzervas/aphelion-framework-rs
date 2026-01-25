//! CubeCL backend integration (feature-gated).

use crate::backend::{Backend, DeviceCapabilities};

#[derive(Debug, Clone)]
pub struct CubeclBackend {
    device: String,
}

impl CubeclBackend {
    pub fn new(device: impl Into<String>) -> Self {
        Self {
            device: device.into(),
        }
    }
}

impl Backend for CubeclBackend {
    fn name(&self) -> &str {
        "cubecl"
    }

    fn device(&self) -> &str {
        &self.device
    }

    fn capabilities(&self) -> DeviceCapabilities {
        // CubeCL is a GPU compute language, so assume GPU-like capabilities
        DeviceCapabilities {
            supports_f16: true,
            supports_bf16: true,
            supports_tf32: false,
            max_memory_bytes: None, // Would be queried from actual device
            compute_units: None,
        }
    }

    fn is_available(&self) -> bool {
        // Placeholder: actual implementation would check device availability
        true
    }
}
