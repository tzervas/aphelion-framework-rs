//! CubeCL backend integration (feature-gated).

use crate::backend::Backend;

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
}
