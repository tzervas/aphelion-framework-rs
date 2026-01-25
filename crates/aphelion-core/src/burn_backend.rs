//! Burn backend integration (feature-gated).

use crate::backend::{Backend, DeviceCapabilities};

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

    fn capabilities(&self) -> DeviceCapabilities {
        match &self.config.device {
            BurnDevice::Cpu => DeviceCapabilities {
                supports_f16: false,
                supports_bf16: false,
                supports_tf32: false,
                max_memory_bytes: None,
                compute_units: None,
            },
            BurnDevice::Cuda(_) => DeviceCapabilities {
                supports_f16: true,
                supports_bf16: true,
                supports_tf32: self.config.allow_tf32,
                max_memory_bytes: None, // Would be queried from actual device
                compute_units: None,
            },
            BurnDevice::Metal(_) => DeviceCapabilities {
                supports_f16: true,
                supports_bf16: false,
                supports_tf32: false,
                max_memory_bytes: None,
                compute_units: None,
            },
            BurnDevice::Vulkan(_) => DeviceCapabilities {
                supports_f16: true,
                supports_bf16: false,
                supports_tf32: false,
                max_memory_bytes: None,
                compute_units: None,
            },
        }
    }

    fn is_available(&self) -> bool {
        // Placeholder: actual implementation would check device availability
        true
    }
}
