//! CubeCL backend integration for the Aphelion framework.
//!
//! This module provides a placeholder implementation for the [CubeCL](https://github.com/tracel-ai/cubecl)
//! GPU compute library. CubeCL is a Rust GPU compute framework that enables writing portable GPU
//! kernels that can target multiple backends including CUDA, ROCm, Metal, Vulkan, and WebGPU.
//!
//! # Feature Flag
//!
//! This module is only available when the `cubecl` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! aphelion-core = { version = "1.1", features = ["cubecl"] }
//! ```
//!
//! # Current Status
//!
//! **This is a placeholder implementation.** The actual CubeCL dependency is not yet integrated.
//! This module provides the API structure and type definitions that will be used when CubeCL
//! integration is completed. The implementation:
//!
//! - Defines device types (`CubeclDevice`) matching CubeCL's supported backends
//! - Provides configuration structures for backend setup
//! - Implements the `Backend` trait with stub methods
//! - Reports CPU as available, GPU backends as unavailable until real integration
//!
//! # CubeCL vs Burn
//!
//! While Burn is a high-level deep learning framework, CubeCL operates at a lower level,
//! providing direct GPU compute capabilities:
//!
//! | Feature | CubeCL | Burn |
//! |---------|--------|------|
//! | Abstraction Level | Low (GPU kernels) | High (neural networks) |
//! | Use Case | Custom GPU algorithms | Deep learning models |
//! | Kernel Definition | Manual | Automatic |
//! | Autodiff | No | Yes |
//!
//! # Future Integration
//!
//! When the CubeCL dependency is enabled, this backend will provide:
//!
//! - Direct GPU compute access via CubeCL's runtime
//! - Custom kernel compilation and execution
//! - Multi-backend support (CUDA, Metal, Vulkan, WebGPU)
//! - Memory management and tensor operations
//! - Integration with Aphelion's pipeline system
//!
//! # Example (Future API)
//!
//! ```ignore
//! use aphelion_core::cubecl_backend::{CubeclBackend, CubeclBackendConfig, CubeclDevice};
//! use aphelion_core::backend::Backend;
//!
//! // Create a GPU-accelerated backend
//! let config = CubeclBackendConfig {
//!     device: CubeclDevice::Cuda(0),
//!     memory_fraction: 0.9,
//! };
//! let mut backend = CubeclBackend::new(config);
//!
//! // Initialize the backend
//! backend.initialize().expect("Failed to initialize CubeCL backend");
//!
//! assert_eq!(backend.name(), "cubecl");
//! assert!(backend.is_available());
//!
//! // Clean up
//! backend.shutdown().expect("Failed to shutdown");
//! ```

use crate::backend::{Backend, DeviceCapabilities};
use crate::error::AphelionResult;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Supported CubeCL device selectors.
///
/// `CubeclDevice` specifies the computational device for CubeCL operations.
/// This enum maps to CubeCL's backend types and will be used for device
/// selection once the CubeCL dependency is integrated.
///
/// # Variants
///
/// - `Cpu` - CPU execution (software fallback, always available)
/// - `Cuda(u32)` - NVIDIA CUDA GPU with device index
/// - `Metal(u32)` - Apple Metal GPU with device index (macOS/iOS only)
/// - `Vulkan(u32)` - Vulkan GPU with device index (cross-platform)
/// - `Wgpu(u32)` - WebGPU backend with device index (cross-platform, web-compatible)
///
/// # Examples
///
/// ```ignore
/// use aphelion_core::cubecl_backend::CubeclDevice;
///
/// let cpu = CubeclDevice::Cpu;
/// let gpu = CubeclDevice::Cuda(0);
///
/// assert_eq!(cpu.as_label(), "cpu");
/// assert_eq!(gpu.as_label(), "cuda:0");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum CubeclDevice {
    /// CPU execution (software fallback, always available)
    #[default]
    Cpu,
    /// NVIDIA CUDA GPU execution with device index
    Cuda(u32),
    /// Apple Metal GPU execution with device index (macOS/iOS only)
    Metal(u32),
    /// Vulkan GPU execution with device index (cross-platform)
    Vulkan(u32),
    /// WebGPU backend with device index (cross-platform, web-compatible)
    Wgpu(u32),
}

impl CubeclDevice {
    /// Returns a human-readable label for this device.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclDevice;
    ///
    /// assert_eq!(CubeclDevice::Cpu.as_label(), "cpu");
    /// assert_eq!(CubeclDevice::Cuda(0).as_label(), "cuda:0");
    /// assert_eq!(CubeclDevice::Metal(1).as_label(), "metal:1");
    /// assert_eq!(CubeclDevice::Vulkan(0).as_label(), "vulkan:0");
    /// assert_eq!(CubeclDevice::Wgpu(0).as_label(), "wgpu:0");
    /// ```
    pub fn as_label(&self) -> String {
        match self {
            CubeclDevice::Cpu => "cpu".to_string(),
            CubeclDevice::Cuda(id) => format!("cuda:{}", id),
            CubeclDevice::Metal(id) => format!("metal:{}", id),
            CubeclDevice::Vulkan(id) => format!("vulkan:{}", id),
            CubeclDevice::Wgpu(id) => format!("wgpu:{}", id),
        }
    }

    /// Returns whether this is a CPU device.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclDevice;
    ///
    /// assert!(CubeclDevice::Cpu.is_cpu());
    /// assert!(!CubeclDevice::Cuda(0).is_cpu());
    /// ```
    pub fn is_cpu(&self) -> bool {
        matches!(self, CubeclDevice::Cpu)
    }

    /// Returns whether this is a GPU device.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclDevice;
    ///
    /// assert!(!CubeclDevice::Cpu.is_gpu());
    /// assert!(CubeclDevice::Cuda(0).is_gpu());
    /// assert!(CubeclDevice::Metal(0).is_gpu());
    /// ```
    pub fn is_gpu(&self) -> bool {
        !self.is_cpu()
    }

    /// Returns whether this device uses NVIDIA CUDA.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclDevice;
    ///
    /// assert!(CubeclDevice::Cuda(0).is_cuda());
    /// assert!(!CubeclDevice::Metal(0).is_cuda());
    /// ```
    pub fn is_cuda(&self) -> bool {
        matches!(self, CubeclDevice::Cuda(_))
    }

    /// Returns whether this device uses Apple Metal.
    pub fn is_metal(&self) -> bool {
        matches!(self, CubeclDevice::Metal(_))
    }

    /// Returns whether this device uses Vulkan.
    pub fn is_vulkan(&self) -> bool {
        matches!(self, CubeclDevice::Vulkan(_))
    }

    /// Returns whether this device uses WebGPU.
    pub fn is_wgpu(&self) -> bool {
        matches!(self, CubeclDevice::Wgpu(_))
    }
}

/// Configuration for the CubeCL backend.
///
/// `CubeclBackendConfig` contains all settings needed to initialize a CubeCL backend,
/// including device selection and memory management options.
///
/// # Examples
///
/// ```ignore
/// use aphelion_core::cubecl_backend::{CubeclBackendConfig, CubeclDevice};
///
/// // Default configuration (CPU)
/// let default_config = CubeclBackendConfig::default();
/// assert!(default_config.device.is_cpu());
///
/// // GPU configuration with memory fraction
/// let gpu_config = CubeclBackendConfig {
///     device: CubeclDevice::Cuda(0),
///     memory_fraction: 0.8,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct CubeclBackendConfig {
    /// The computational device to use
    pub device: CubeclDevice,
    /// Fraction of GPU memory to allocate (0.0 to 1.0)
    /// This limits memory usage to prevent OOM errors when sharing GPU with other processes
    pub memory_fraction: f32,
}

impl Default for CubeclBackendConfig {
    fn default() -> Self {
        Self {
            device: CubeclDevice::Cpu,
            memory_fraction: 0.9,
        }
    }
}

impl CubeclBackendConfig {
    /// Creates a new configuration with the specified device.
    ///
    /// # Arguments
    ///
    /// * `device` - The computational device to use
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::{CubeclBackendConfig, CubeclDevice};
    ///
    /// let config = CubeclBackendConfig::new(CubeclDevice::Cuda(0));
    /// assert_eq!(config.memory_fraction, 0.9);
    /// ```
    pub fn new(device: CubeclDevice) -> Self {
        Self {
            device,
            memory_fraction: 0.9,
        }
    }

    /// Creates a CPU configuration.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclBackendConfig;
    ///
    /// let config = CubeclBackendConfig::cpu();
    /// assert!(config.device.is_cpu());
    /// ```
    pub fn cpu() -> Self {
        Self::new(CubeclDevice::Cpu)
    }

    /// Creates a CUDA GPU configuration.
    ///
    /// # Arguments
    ///
    /// * `device_id` - CUDA device index (0 for first GPU)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclBackendConfig;
    ///
    /// let config = CubeclBackendConfig::cuda(0);
    /// assert!(config.device.is_gpu());
    /// ```
    pub fn cuda(device_id: u32) -> Self {
        Self::new(CubeclDevice::Cuda(device_id))
    }

    /// Creates a Metal GPU configuration.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Metal device index (0 for first GPU)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclBackendConfig;
    ///
    /// let config = CubeclBackendConfig::metal(0);
    /// assert!(config.device.is_metal());
    /// ```
    pub fn metal(device_id: u32) -> Self {
        Self::new(CubeclDevice::Metal(device_id))
    }

    /// Creates a Vulkan GPU configuration.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Vulkan device index (0 for first GPU)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclBackendConfig;
    ///
    /// let config = CubeclBackendConfig::vulkan(0);
    /// assert!(config.device.is_vulkan());
    /// ```
    pub fn vulkan(device_id: u32) -> Self {
        Self::new(CubeclDevice::Vulkan(device_id))
    }

    /// Creates a WebGPU configuration.
    ///
    /// # Arguments
    ///
    /// * `device_id` - WGPU device index (0 for first adapter)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclBackendConfig;
    ///
    /// let config = CubeclBackendConfig::wgpu(0);
    /// assert!(config.device.is_wgpu());
    /// ```
    pub fn wgpu(device_id: u32) -> Self {
        Self::new(CubeclDevice::Wgpu(device_id))
    }

    /// Sets the memory fraction for GPU memory allocation.
    ///
    /// # Arguments
    ///
    /// * `fraction` - Fraction of GPU memory to use (0.0 to 1.0)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclBackendConfig;
    ///
    /// let config = CubeclBackendConfig::cuda(0).with_memory_fraction(0.5);
    /// assert_eq!(config.memory_fraction, 0.5);
    /// ```
    pub fn with_memory_fraction(mut self, fraction: f32) -> Self {
        self.memory_fraction = fraction.clamp(0.0, 1.0);
        self
    }
}

/// CubeCL backend implementation for the Aphelion framework.
///
/// `CubeclBackend` provides integration with the CubeCL GPU compute library.
/// This is currently a placeholder implementation that will be connected to
/// the actual CubeCL library when the dependency is enabled.
///
/// # Thread Safety
///
/// `CubeclBackend` is thread-safe and can be shared across threads. The internal
/// state uses atomic operations for the initialization flag.
///
/// # Examples
///
/// ```ignore
/// use aphelion_core::cubecl_backend::{CubeclBackend, CubeclBackendConfig, CubeclDevice};
/// use aphelion_core::backend::Backend;
///
/// let backend = CubeclBackend::new(CubeclBackendConfig::default());
/// assert_eq!(backend.name(), "cubecl");
/// assert_eq!(backend.device(), "cpu");
/// ```
#[derive(Debug)]
pub struct CubeclBackend {
    config: CubeclBackendConfig,
    initialized: Arc<AtomicBool>,
}

impl Clone for CubeclBackend {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            initialized: Arc::new(AtomicBool::new(self.initialized.load(Ordering::SeqCst))),
        }
    }
}

impl CubeclBackend {
    /// Creates a new CubeCL backend with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Backend configuration
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::{CubeclBackend, CubeclBackendConfig};
    ///
    /// let backend = CubeclBackend::new(CubeclBackendConfig::default());
    /// ```
    pub fn new(config: CubeclBackendConfig) -> Self {
        Self {
            config,
            initialized: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Creates a CPU backend with default configuration.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclBackend;
    /// use aphelion_core::backend::Backend;
    ///
    /// let backend = CubeclBackend::cpu();
    /// assert_eq!(backend.device(), "cpu");
    /// ```
    pub fn cpu() -> Self {
        Self::new(CubeclBackendConfig::cpu())
    }

    /// Creates a CUDA GPU backend.
    ///
    /// # Arguments
    ///
    /// * `device_id` - CUDA device index (0 for first GPU)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclBackend;
    /// use aphelion_core::backend::Backend;
    ///
    /// let backend = CubeclBackend::cuda(0);
    /// assert_eq!(backend.device(), "cuda");
    /// ```
    pub fn cuda(device_id: u32) -> Self {
        Self::new(CubeclBackendConfig::cuda(device_id))
    }

    /// Creates a Metal GPU backend.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Metal device index (0 for first GPU)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclBackend;
    /// use aphelion_core::backend::Backend;
    ///
    /// let backend = CubeclBackend::metal(0);
    /// assert_eq!(backend.device(), "metal");
    /// ```
    pub fn metal(device_id: u32) -> Self {
        Self::new(CubeclBackendConfig::metal(device_id))
    }

    /// Creates a Vulkan GPU backend.
    ///
    /// # Arguments
    ///
    /// * `device_id` - Vulkan device index (0 for first GPU)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclBackend;
    /// use aphelion_core::backend::Backend;
    ///
    /// let backend = CubeclBackend::vulkan(0);
    /// assert_eq!(backend.device(), "vulkan");
    /// ```
    pub fn vulkan(device_id: u32) -> Self {
        Self::new(CubeclBackendConfig::vulkan(device_id))
    }

    /// Creates a WebGPU backend.
    ///
    /// # Arguments
    ///
    /// * `device_id` - WGPU device index (0 for first adapter)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclBackend;
    /// use aphelion_core::backend::Backend;
    ///
    /// let backend = CubeclBackend::wgpu(0);
    /// assert_eq!(backend.device(), "wgpu");
    /// ```
    pub fn wgpu(device_id: u32) -> Self {
        Self::new(CubeclBackendConfig::wgpu(device_id))
    }

    /// Returns the backend configuration.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::{CubeclBackend, CubeclBackendConfig, CubeclDevice};
    ///
    /// let backend = CubeclBackend::cpu();
    /// assert!(backend.config().device.is_cpu());
    /// ```
    pub fn config(&self) -> &CubeclBackendConfig {
        &self.config
    }

    /// Returns whether the backend has been initialized.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::cubecl_backend::CubeclBackend;
    ///
    /// let backend = CubeclBackend::cpu();
    /// assert!(!backend.is_initialized());
    /// ```
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::SeqCst)
    }

    /// Checks if the configured device is available on this system.
    ///
    /// # Note
    ///
    /// In the placeholder implementation:
    /// - CPU is always available
    /// - CUDA reports unavailable (no actual CUDA detection yet)
    /// - Metal reports available only on macOS
    /// - Vulkan reports unavailable (no actual Vulkan detection yet)
    /// - WGPU reports unavailable (no actual WGPU detection yet)
    fn check_device_availability(&self) -> bool {
        match &self.config.device {
            CubeclDevice::Cpu => true,
            CubeclDevice::Cuda(_) => {
                // In actual implementation, this would check CUDA availability
                // via CubeCL's device detection. For now, report as unavailable.
                false
            }
            CubeclDevice::Metal(_) => {
                // Metal is only available on macOS/iOS
                cfg!(target_os = "macos")
            }
            CubeclDevice::Vulkan(_) => {
                // In actual implementation, this would check Vulkan availability.
                // For now, report as unavailable.
                false
            }
            CubeclDevice::Wgpu(_) => {
                // In actual implementation, this would check WGPU adapter availability.
                // For now, report as unavailable.
                false
            }
        }
    }
}

impl Default for CubeclBackend {
    fn default() -> Self {
        Self::cpu()
    }
}

impl Backend for CubeclBackend {
    fn name(&self) -> &str {
        "cubecl"
    }

    fn device(&self) -> &str {
        match &self.config.device {
            CubeclDevice::Cpu => "cpu",
            CubeclDevice::Cuda(_) => "cuda",
            CubeclDevice::Metal(_) => "metal",
            CubeclDevice::Vulkan(_) => "vulkan",
            CubeclDevice::Wgpu(_) => "wgpu",
        }
    }

    fn capabilities(&self) -> DeviceCapabilities {
        match &self.config.device {
            CubeclDevice::Cpu => DeviceCapabilities {
                supports_f16: false,
                supports_bf16: false,
                supports_tf32: false,
                max_memory_bytes: None,
                compute_units: None,
            },
            CubeclDevice::Cuda(_) => DeviceCapabilities {
                supports_f16: true,
                supports_bf16: true,
                supports_tf32: true,
                // Placeholder values - actual implementation would query the device
                max_memory_bytes: Some(8 * 1024 * 1024 * 1024), // 8 GB placeholder
                compute_units: Some(128),                       // Placeholder
            },
            CubeclDevice::Metal(_) => DeviceCapabilities {
                supports_f16: true,
                supports_bf16: false,
                supports_tf32: false,
                max_memory_bytes: None,
                compute_units: None,
            },
            CubeclDevice::Vulkan(_) => DeviceCapabilities {
                supports_f16: true,
                supports_bf16: false,
                supports_tf32: false,
                max_memory_bytes: None,
                compute_units: None,
            },
            CubeclDevice::Wgpu(_) => DeviceCapabilities {
                supports_f16: true, // WebGPU typically supports f16
                supports_bf16: false,
                supports_tf32: false,
                max_memory_bytes: None,
                compute_units: None,
            },
        }
    }

    fn is_available(&self) -> bool {
        self.check_device_availability()
    }

    fn initialize(&mut self) -> AphelionResult<()> {
        if self.initialized.load(Ordering::SeqCst) {
            return Ok(());
        }

        if !self.check_device_availability() {
            return Err(crate::error::AphelionError::backend(format!(
                "CubeCL device {} is not available",
                self.config.device.as_label()
            )));
        }

        // In actual implementation, this would initialize the CubeCL runtime:
        // - Create the device handle via CubeCL
        // - Set up memory allocators
        // - Configure memory limits based on memory_fraction
        // For now, just mark as initialized
        self.initialized.store(true, Ordering::SeqCst);

        tracing::info!(
            backend = "cubecl",
            device = %self.config.device.as_label(),
            memory_fraction = self.config.memory_fraction,
            "CubeCL backend initialized (placeholder)"
        );

        Ok(())
    }

    fn shutdown(&mut self) -> AphelionResult<()> {
        if !self.initialized.load(Ordering::SeqCst) {
            return Ok(());
        }

        // In actual implementation, this would:
        // - Release device handles
        // - Free allocated memory
        // - Synchronize pending operations
        self.initialized.store(false, Ordering::SeqCst);

        tracing::info!(
            backend = "cubecl",
            device = %self.config.device.as_label(),
            "CubeCL backend shutdown (placeholder)"
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // CubeclDevice Tests
    // ==========================================================================

    #[test]
    fn test_cubecl_device_as_label() {
        assert_eq!(CubeclDevice::Cpu.as_label(), "cpu");
        assert_eq!(CubeclDevice::Cuda(0).as_label(), "cuda:0");
        assert_eq!(CubeclDevice::Cuda(1).as_label(), "cuda:1");
        assert_eq!(CubeclDevice::Metal(0).as_label(), "metal:0");
        assert_eq!(CubeclDevice::Vulkan(2).as_label(), "vulkan:2");
        assert_eq!(CubeclDevice::Wgpu(0).as_label(), "wgpu:0");
    }

    #[test]
    fn test_cubecl_device_default() {
        let device = CubeclDevice::default();
        assert_eq!(device, CubeclDevice::Cpu);
    }

    #[test]
    fn test_cubecl_device_is_cpu_gpu() {
        assert!(CubeclDevice::Cpu.is_cpu());
        assert!(!CubeclDevice::Cpu.is_gpu());

        assert!(!CubeclDevice::Cuda(0).is_cpu());
        assert!(CubeclDevice::Cuda(0).is_gpu());

        assert!(!CubeclDevice::Metal(0).is_cpu());
        assert!(CubeclDevice::Metal(0).is_gpu());

        assert!(!CubeclDevice::Vulkan(0).is_cpu());
        assert!(CubeclDevice::Vulkan(0).is_gpu());

        assert!(!CubeclDevice::Wgpu(0).is_cpu());
        assert!(CubeclDevice::Wgpu(0).is_gpu());
    }

    #[test]
    fn test_cubecl_device_type_checks() {
        assert!(CubeclDevice::Cuda(0).is_cuda());
        assert!(!CubeclDevice::Metal(0).is_cuda());

        assert!(CubeclDevice::Metal(0).is_metal());
        assert!(!CubeclDevice::Cuda(0).is_metal());

        assert!(CubeclDevice::Vulkan(0).is_vulkan());
        assert!(!CubeclDevice::Metal(0).is_vulkan());

        assert!(CubeclDevice::Wgpu(0).is_wgpu());
        assert!(!CubeclDevice::Vulkan(0).is_wgpu());
    }

    #[test]
    fn test_cubecl_device_clone() {
        let device1 = CubeclDevice::Cuda(1);
        let device2 = device1.clone();
        assert_eq!(device1, device2);
    }

    // ==========================================================================
    // CubeclBackendConfig Tests
    // ==========================================================================

    #[test]
    fn test_cubecl_backend_config_default() {
        let config = CubeclBackendConfig::default();
        assert_eq!(config.device, CubeclDevice::Cpu);
        assert!((config.memory_fraction - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cubecl_backend_config_new() {
        let config = CubeclBackendConfig::new(CubeclDevice::Cuda(1));
        assert_eq!(config.device, CubeclDevice::Cuda(1));
        assert!((config.memory_fraction - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cubecl_backend_config_cpu() {
        let config = CubeclBackendConfig::cpu();
        assert!(config.device.is_cpu());
    }

    #[test]
    fn test_cubecl_backend_config_cuda() {
        let config = CubeclBackendConfig::cuda(0);
        assert_eq!(config.device, CubeclDevice::Cuda(0));
    }

    #[test]
    fn test_cubecl_backend_config_metal() {
        let config = CubeclBackendConfig::metal(0);
        assert_eq!(config.device, CubeclDevice::Metal(0));
    }

    #[test]
    fn test_cubecl_backend_config_vulkan() {
        let config = CubeclBackendConfig::vulkan(0);
        assert_eq!(config.device, CubeclDevice::Vulkan(0));
    }

    #[test]
    fn test_cubecl_backend_config_wgpu() {
        let config = CubeclBackendConfig::wgpu(0);
        assert_eq!(config.device, CubeclDevice::Wgpu(0));
    }

    #[test]
    fn test_cubecl_backend_config_with_memory_fraction() {
        let config = CubeclBackendConfig::cuda(0).with_memory_fraction(0.5);
        assert!((config.memory_fraction - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cubecl_backend_config_memory_fraction_clamped() {
        let config_low = CubeclBackendConfig::cuda(0).with_memory_fraction(-0.5);
        assert!((config_low.memory_fraction - 0.0).abs() < f32::EPSILON);

        let config_high = CubeclBackendConfig::cuda(0).with_memory_fraction(1.5);
        assert!((config_high.memory_fraction - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cubecl_backend_config_builder_chain() {
        let config = CubeclBackendConfig::cuda(0).with_memory_fraction(0.7);
        assert_eq!(config.device, CubeclDevice::Cuda(0));
        assert!((config.memory_fraction - 0.7).abs() < f32::EPSILON);
    }

    // ==========================================================================
    // CubeclBackend Tests
    // ==========================================================================

    #[test]
    fn test_cubecl_backend_new() {
        let backend = CubeclBackend::new(CubeclBackendConfig::default());
        assert_eq!(backend.name(), "cubecl");
        assert_eq!(backend.device(), "cpu");
        assert!(!backend.is_initialized());
    }

    #[test]
    fn test_cubecl_backend_cpu() {
        let backend = CubeclBackend::cpu();
        assert_eq!(backend.device(), "cpu");
        assert!(backend.config().device.is_cpu());
    }

    #[test]
    fn test_cubecl_backend_cuda() {
        let backend = CubeclBackend::cuda(0);
        assert_eq!(backend.device(), "cuda");
        assert!(backend.config().device.is_gpu());
    }

    #[test]
    fn test_cubecl_backend_metal() {
        let backend = CubeclBackend::metal(0);
        assert_eq!(backend.device(), "metal");
        assert!(backend.config().device.is_metal());
    }

    #[test]
    fn test_cubecl_backend_vulkan() {
        let backend = CubeclBackend::vulkan(0);
        assert_eq!(backend.device(), "vulkan");
        assert!(backend.config().device.is_vulkan());
    }

    #[test]
    fn test_cubecl_backend_wgpu() {
        let backend = CubeclBackend::wgpu(0);
        assert_eq!(backend.device(), "wgpu");
        assert!(backend.config().device.is_wgpu());
    }

    #[test]
    fn test_cubecl_backend_default() {
        let backend = CubeclBackend::default();
        assert_eq!(backend.device(), "cpu");
    }

    #[test]
    fn test_cubecl_backend_capabilities_cpu() {
        let backend = CubeclBackend::cpu();
        let caps = backend.capabilities();
        assert!(!caps.supports_f16);
        assert!(!caps.supports_bf16);
        assert!(!caps.supports_tf32);
        assert!(caps.max_memory_bytes.is_none());
    }

    #[test]
    fn test_cubecl_backend_capabilities_cuda() {
        let backend = CubeclBackend::cuda(0);
        let caps = backend.capabilities();
        assert!(caps.supports_f16);
        assert!(caps.supports_bf16);
        assert!(caps.supports_tf32);
        assert!(caps.max_memory_bytes.is_some());
    }

    #[test]
    fn test_cubecl_backend_capabilities_metal() {
        let backend = CubeclBackend::metal(0);
        let caps = backend.capabilities();
        assert!(caps.supports_f16);
        assert!(!caps.supports_bf16);
        assert!(!caps.supports_tf32);
    }

    #[test]
    fn test_cubecl_backend_capabilities_wgpu() {
        let backend = CubeclBackend::wgpu(0);
        let caps = backend.capabilities();
        assert!(caps.supports_f16);
        assert!(!caps.supports_bf16);
        assert!(!caps.supports_tf32);
    }

    #[test]
    fn test_cubecl_backend_is_available_cpu() {
        let backend = CubeclBackend::cpu();
        assert!(backend.is_available());
    }

    #[test]
    fn test_cubecl_backend_is_available_cuda() {
        let backend = CubeclBackend::cuda(0);
        // CUDA is not available in placeholder implementation
        assert!(!backend.is_available());
    }

    #[test]
    fn test_cubecl_backend_is_available_vulkan() {
        let backend = CubeclBackend::vulkan(0);
        // Vulkan is not available in placeholder implementation
        assert!(!backend.is_available());
    }

    #[test]
    fn test_cubecl_backend_is_available_wgpu() {
        let backend = CubeclBackend::wgpu(0);
        // WGPU is not available in placeholder implementation
        assert!(!backend.is_available());
    }

    #[test]
    fn test_cubecl_backend_initialize_cpu() {
        let mut backend = CubeclBackend::cpu();
        assert!(!backend.is_initialized());

        let result = backend.initialize();
        assert!(result.is_ok());
        assert!(backend.is_initialized());

        // Double initialization should be ok
        let result = backend.initialize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_cubecl_backend_initialize_cuda_fails() {
        let mut backend = CubeclBackend::cuda(0);
        let result = backend.initialize();
        // Should fail because CUDA is not available in placeholder
        assert!(result.is_err());
    }

    #[test]
    fn test_cubecl_backend_initialize_vulkan_fails() {
        let mut backend = CubeclBackend::vulkan(0);
        let result = backend.initialize();
        // Should fail because Vulkan is not available in placeholder
        assert!(result.is_err());
    }

    #[test]
    fn test_cubecl_backend_initialize_wgpu_fails() {
        let mut backend = CubeclBackend::wgpu(0);
        let result = backend.initialize();
        // Should fail because WGPU is not available in placeholder
        assert!(result.is_err());
    }

    #[test]
    fn test_cubecl_backend_shutdown() {
        let mut backend = CubeclBackend::cpu();
        backend.initialize().unwrap();
        assert!(backend.is_initialized());

        let result = backend.shutdown();
        assert!(result.is_ok());
        assert!(!backend.is_initialized());

        // Double shutdown should be ok
        let result = backend.shutdown();
        assert!(result.is_ok());
    }

    #[test]
    fn test_cubecl_backend_shutdown_without_init() {
        let mut backend = CubeclBackend::cpu();
        assert!(!backend.is_initialized());

        // Shutdown without init should be ok
        let result = backend.shutdown();
        assert!(result.is_ok());
    }

    #[test]
    fn test_cubecl_backend_clone() {
        let mut backend = CubeclBackend::cpu();
        backend.initialize().unwrap();

        let cloned = backend.clone();
        // Cloned backend has its own initialization state
        assert!(cloned.is_initialized());
        assert_eq!(cloned.device(), backend.device());
    }

    #[test]
    fn test_cubecl_backend_clone_preserves_config() {
        let backend = CubeclBackend::new(CubeclBackendConfig::cuda(1).with_memory_fraction(0.5));
        let cloned = backend.clone();

        assert_eq!(cloned.config().device, CubeclDevice::Cuda(1));
        assert!((cloned.config().memory_fraction - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cubecl_backend_lifecycle() {
        let mut backend = CubeclBackend::cpu();

        // Initial state
        assert!(!backend.is_initialized());
        assert!(backend.is_available());

        // Initialize
        let init_result = backend.initialize();
        assert!(init_result.is_ok());
        assert!(backend.is_initialized());

        // Shutdown
        let shutdown_result = backend.shutdown();
        assert!(shutdown_result.is_ok());
        assert!(!backend.is_initialized());

        // Re-initialize should work
        let reinit_result = backend.initialize();
        assert!(reinit_result.is_ok());
        assert!(backend.is_initialized());
    }

    #[test]
    fn test_cubecl_backend_config_access() {
        let backend = CubeclBackend::new(CubeclBackendConfig::cuda(2).with_memory_fraction(0.75));

        let config = backend.config();
        assert_eq!(config.device, CubeclDevice::Cuda(2));
        assert!((config.memory_fraction - 0.75).abs() < f32::EPSILON);
    }
}
