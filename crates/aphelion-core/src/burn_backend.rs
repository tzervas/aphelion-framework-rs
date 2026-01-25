//! Burn backend integration (feature-gated).

use crate::backend::Backend;

/// Supported burn device selectors (placeholder).
#[derive(Debug, Clone)]
pub enum BurnDevice {
    Cpu,
    Cuda(u32),
    Metal(u32),
    Vulkan(u32),
}

impl BurnDevice {
    pub fn as_label(&self) -> String {
        match self {
            BurnDevice::Cpu => "cpu".to_string(),
            BurnDevice::Cuda(id) => format!("cuda:{}", id),
            BurnDevice::Metal(id) => format!("metal:{}", id),
            BurnDevice::Vulkan(id) => format!("vulkan:{}", id),
        }
    }
}

/// Burn backend configuration (placeholder).
#[derive(Debug, Clone)]
pub struct BurnBackendConfig {
    pub device: BurnDevice,
    pub allow_tf32: bool,
}

impl Default for BurnBackendConfig {
    fn default() -> Self {
        Self {
            device: BurnDevice::Cpu,
            allow_tf32: false,
        }
    }
}

/// Simple burn backend stub that carries device configuration.
#[derive(Debug, Clone)]
pub struct BurnBackend {
    config: BurnBackendConfig,
}

impl BurnBackend {
    pub fn new(config: BurnBackendConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &BurnBackendConfig {
        &self.config
    }
}

impl Backend for BurnBackend {
    fn name(&self) -> &str {
        "burn"
    }

    fn device(&self) -> &str {
        // NOTE: This returns a cached label for now.
        // Future: map to actual burn device handle.
        match &self.config.device {
            BurnDevice::Cpu => "cpu",
            BurnDevice::Cuda(_) => "cuda",
            BurnDevice::Metal(_) => "metal",
            BurnDevice::Vulkan(_) => "vulkan",
        }
    }
}
