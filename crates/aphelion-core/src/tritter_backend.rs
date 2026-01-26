//! Tritter-accel backend integration for accelerated training and inference.
//!
//! This module provides integration with the tritter-accel library for hardware-accelerated
//! AI operations including gradient compression, ternary quantization, and optimized inference.
//!
//! # Feature Flag
//!
//! This module is only available when the `tritter-accel` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! aphelion-core = { version = "1.1", features = ["tritter-accel"] }
//! ```
//!
//! # Acceleration Modes
//!
//! The backend supports two primary modes:
//!
//! - **Training Mode**: Gradient compression (10-100x), mixed precision, deterministic training
//! - **Inference Mode**: Ternary layer conversion, KV caching, batched execution
//!
//! # Examples
//!
//! ```ignore
//! use aphelion_core::tritter_backend::{TriterAccelBackend, TriterDevice};
//!
//! // Create a GPU-accelerated backend
//! let backend = TriterAccelBackend::new(TriterDevice::Cuda(0))
//!     .with_training_mode(0.1)  // 10x gradient compression
//!     .expect("Failed to initialize backend");
//!
//! assert_eq!(backend.name(), "tritter-accel");
//! assert!(backend.is_available());
//! ```

use crate::backend::{Backend, DeviceCapabilities};
use crate::error::{AphelionError, AphelionResult};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Device types supported by tritter-accel.
///
/// `TriterDevice` specifies the computational device for acceleration operations.
/// The actual availability depends on hardware and driver support.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TriterDevice {
    /// CPU execution (always available)
    #[default]
    Cpu,
    /// CUDA GPU execution with device index
    Cuda(usize),
    /// Metal GPU execution (macOS only)
    Metal,
}

impl TriterDevice {
    /// Returns the device identifier string.
    pub fn as_str(&self) -> String {
        match self {
            TriterDevice::Cpu => "cpu".to_string(),
            TriterDevice::Cuda(idx) => format!("cuda:{}", idx),
            TriterDevice::Metal => "metal".to_string(),
        }
    }
}

/// Configuration for training mode acceleration.
///
/// Training mode enables gradient compression using VSA (Vector Symbolic Architecture)
/// for efficient distributed training with reduced communication overhead.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Compression ratio for gradients (0.01 = 100x, 0.1 = 10x, 1.0 = no compression)
    pub compression_ratio: f32,
    /// Enable deterministic phase training for reproducibility
    pub deterministic: bool,
    /// Seed for deterministic operations
    pub seed: Option<u64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            compression_ratio: 0.1, // 10x compression by default
            deterministic: true,
            seed: None,
        }
    }
}

impl TrainingConfig {
    /// Creates a new training configuration with the specified compression ratio.
    ///
    /// # Arguments
    ///
    /// * `compression_ratio` - Gradient compression ratio (0.01-1.0)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::tritter_backend::TrainingConfig;
    ///
    /// let config = TrainingConfig::new(0.1); // 10x compression
    /// ```
    pub fn new(compression_ratio: f32) -> Self {
        Self {
            compression_ratio: compression_ratio.clamp(0.01, 1.0),
            ..Default::default()
        }
    }

    /// Sets the deterministic mode.
    pub fn with_deterministic(mut self, deterministic: bool) -> Self {
        self.deterministic = deterministic;
        self
    }

    /// Sets the seed for deterministic operations.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Configuration for inference mode acceleration.
///
/// Inference mode enables optimizations like ternary layer conversion
/// for memory-efficient inference and KV caching for autoregressive models.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Enable ternary layer conversion for 16x memory reduction
    pub use_ternary_layers: bool,
    /// Batch size for batched inference
    pub batch_size: usize,
    /// Enable KV caching for autoregressive inference
    pub use_kv_cache: bool,
    /// Maximum sequence length for KV cache
    pub max_seq_len: Option<usize>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            use_ternary_layers: true,
            batch_size: 1,
            use_kv_cache: false,
            max_seq_len: None,
        }
    }
}

impl InferenceConfig {
    /// Creates a new inference configuration with the specified batch size.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of samples to process in parallel
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::tritter_backend::InferenceConfig;
    ///
    /// let config = InferenceConfig::new(32); // Batch size of 32
    /// ```
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size: batch_size.max(1),
            ..Default::default()
        }
    }

    /// Enables or disables ternary layer conversion.
    pub fn with_ternary_layers(mut self, enabled: bool) -> Self {
        self.use_ternary_layers = enabled;
        self
    }

    /// Enables KV caching with the specified maximum sequence length.
    pub fn with_kv_cache(mut self, max_seq_len: usize) -> Self {
        self.use_kv_cache = true;
        self.max_seq_len = Some(max_seq_len);
        self
    }
}

/// Acceleration mode for the tritter-accel backend.
#[derive(Debug, Clone, Default)]
pub enum AccelerationMode {
    /// No acceleration (passthrough mode)
    #[default]
    None,
    /// Training mode with gradient compression
    Training(TrainingConfig),
    /// Inference mode with optimizations
    Inference(InferenceConfig),
}

/// Backend implementation for tritter-accel acceleration.
///
/// `TriterAccelBackend` provides integration with the tritter-accel library for
/// hardware-accelerated training and inference operations. It supports gradient
/// compression, ternary quantization, and optimized inference paths.
///
/// # Examples
///
/// ```ignore
/// use aphelion_core::tritter_backend::{TriterAccelBackend, TriterDevice};
/// use aphelion_core::backend::Backend;
///
/// let backend = TriterAccelBackend::new(TriterDevice::Cpu);
/// assert_eq!(backend.name(), "tritter-accel");
/// assert!(backend.is_available());
/// ```
#[derive(Debug)]
pub struct TriterAccelBackend {
    device: TriterDevice,
    mode: AccelerationMode,
    initialized: Arc<AtomicBool>,
}

impl Clone for TriterAccelBackend {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            mode: self.mode.clone(),
            initialized: Arc::new(AtomicBool::new(self.initialized.load(Ordering::SeqCst))),
        }
    }
}

impl TriterAccelBackend {
    /// Creates a new tritter-accel backend with the specified device.
    ///
    /// # Arguments
    ///
    /// * `device` - The computational device to use
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::tritter_backend::{TriterAccelBackend, TriterDevice};
    ///
    /// let cpu_backend = TriterAccelBackend::new(TriterDevice::Cpu);
    /// let gpu_backend = TriterAccelBackend::new(TriterDevice::Cuda(0));
    /// ```
    pub fn new(device: TriterDevice) -> Self {
        Self {
            device,
            mode: AccelerationMode::None,
            initialized: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Creates a CPU backend.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::tritter_backend::TriterAccelBackend;
    ///
    /// let backend = TriterAccelBackend::cpu();
    /// ```
    pub fn cpu() -> Self {
        Self::new(TriterDevice::Cpu)
    }

    /// Creates a CUDA GPU backend with the specified device index.
    ///
    /// # Arguments
    ///
    /// * `device_idx` - CUDA device index (0 for first GPU)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::tritter_backend::TriterAccelBackend;
    ///
    /// let backend = TriterAccelBackend::cuda(0);
    /// ```
    pub fn cuda(device_idx: usize) -> Self {
        Self::new(TriterDevice::Cuda(device_idx))
    }

    /// Creates a Metal GPU backend (macOS only).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::tritter_backend::TriterAccelBackend;
    ///
    /// let backend = TriterAccelBackend::metal();
    /// ```
    pub fn metal() -> Self {
        Self::new(TriterDevice::Metal)
    }

    /// Configures the backend for training mode with gradient compression.
    ///
    /// # Arguments
    ///
    /// * `compression_ratio` - Gradient compression ratio (0.01-1.0)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::tritter_backend::TriterAccelBackend;
    ///
    /// let backend = TriterAccelBackend::cpu()
    ///     .with_training_mode(0.1);  // 10x compression
    /// ```
    pub fn with_training_mode(mut self, compression_ratio: f32) -> Self {
        self.mode = AccelerationMode::Training(TrainingConfig::new(compression_ratio));
        self
    }

    /// Configures the backend for training mode with full configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Training configuration
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::tritter_backend::{TriterAccelBackend, TrainingConfig};
    ///
    /// let config = TrainingConfig::new(0.1)
    ///     .with_deterministic(true)
    ///     .with_seed(42);
    ///
    /// let backend = TriterAccelBackend::cpu()
    ///     .with_training_config(config);
    /// ```
    pub fn with_training_config(mut self, config: TrainingConfig) -> Self {
        self.mode = AccelerationMode::Training(config);
        self
    }

    /// Configures the backend for inference mode.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Batch size for inference
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::tritter_backend::TriterAccelBackend;
    ///
    /// let backend = TriterAccelBackend::cpu()
    ///     .with_inference_mode(32);  // Batch size 32
    /// ```
    pub fn with_inference_mode(mut self, batch_size: usize) -> Self {
        self.mode = AccelerationMode::Inference(InferenceConfig::new(batch_size));
        self
    }

    /// Configures the backend for inference mode with full configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Inference configuration
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::tritter_backend::{TriterAccelBackend, InferenceConfig};
    ///
    /// let config = InferenceConfig::new(32)
    ///     .with_ternary_layers(true)
    ///     .with_kv_cache(2048);
    ///
    /// let backend = TriterAccelBackend::cpu()
    ///     .with_inference_config(config);
    /// ```
    pub fn with_inference_config(mut self, config: InferenceConfig) -> Self {
        self.mode = AccelerationMode::Inference(config);
        self
    }

    /// Returns the current acceleration mode.
    pub fn acceleration_mode(&self) -> &AccelerationMode {
        &self.mode
    }

    /// Returns whether the backend is in training mode.
    pub fn is_training_mode(&self) -> bool {
        matches!(self.mode, AccelerationMode::Training(_))
    }

    /// Returns whether the backend is in inference mode.
    pub fn is_inference_mode(&self) -> bool {
        matches!(self.mode, AccelerationMode::Inference(_))
    }

    /// Returns the training configuration if in training mode.
    pub fn training_config(&self) -> Option<&TrainingConfig> {
        match &self.mode {
            AccelerationMode::Training(config) => Some(config),
            _ => None,
        }
    }

    /// Returns the inference configuration if in inference mode.
    pub fn inference_config(&self) -> Option<&InferenceConfig> {
        match &self.mode {
            AccelerationMode::Inference(config) => Some(config),
            _ => None,
        }
    }

    /// Returns whether the backend has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::SeqCst)
    }

    /// Checks if the specified device is actually available on the system.
    ///
    /// Note: In actual implementation with tritter-accel dependency,
    /// this would query the actual hardware. For now, CPU is always
    /// available, and GPU devices report unavailable until the actual
    /// tritter-accel integration is connected.
    fn check_device_availability(&self) -> bool {
        match &self.device {
            TriterDevice::Cpu => true,
            TriterDevice::Cuda(_) => {
                // In actual implementation, this would check CUDA availability
                // via tritter-accel. For now, report as unavailable.
                false
            }
            TriterDevice::Metal => {
                // Metal is only available on macOS
                cfg!(target_os = "macos")
            }
        }
    }
}

impl Backend for TriterAccelBackend {
    fn name(&self) -> &str {
        "tritter-accel"
    }

    fn device(&self) -> &str {
        match &self.device {
            TriterDevice::Cpu => "cpu",
            TriterDevice::Cuda(idx) => {
                // Return a static string for common cases
                match idx {
                    0 => "cuda:0",
                    1 => "cuda:1",
                    2 => "cuda:2",
                    3 => "cuda:3",
                    _ => "cuda:n",
                }
            }
            TriterDevice::Metal => "metal",
        }
    }

    fn capabilities(&self) -> DeviceCapabilities {
        match &self.device {
            TriterDevice::Cpu => DeviceCapabilities {
                supports_f16: false,
                supports_bf16: false,
                supports_tf32: false,
                max_memory_bytes: None,
                compute_units: None,
            },
            TriterDevice::Cuda(_) => DeviceCapabilities {
                supports_f16: true,
                supports_bf16: true,
                supports_tf32: true,
                // Actual values would be queried from the device
                max_memory_bytes: Some(8 * 1024 * 1024 * 1024), // 8 GB placeholder
                compute_units: Some(128),                       // Placeholder
            },
            TriterDevice::Metal => DeviceCapabilities {
                supports_f16: true,
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
            return Err(AphelionError::backend(format!(
                "Device {} is not available",
                self.device.as_str()
            )));
        }

        // In actual implementation, this would initialize tritter-accel
        // For now, just mark as initialized
        self.initialized.store(true, Ordering::SeqCst);

        tracing::info!(
            backend = "tritter-accel",
            device = %self.device.as_str(),
            mode = ?self.mode,
            "Backend initialized"
        );

        Ok(())
    }

    fn shutdown(&mut self) -> AphelionResult<()> {
        if !self.initialized.load(Ordering::SeqCst) {
            return Ok(());
        }

        // In actual implementation, this would clean up tritter-accel resources
        self.initialized.store(false, Ordering::SeqCst);

        tracing::info!(
            backend = "tritter-accel",
            device = %self.device.as_str(),
            "Backend shutdown"
        );

        Ok(())
    }
}

impl Default for TriterAccelBackend {
    fn default() -> Self {
        Self::cpu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triter_device_as_str() {
        assert_eq!(TriterDevice::Cpu.as_str(), "cpu");
        assert_eq!(TriterDevice::Cuda(0).as_str(), "cuda:0");
        assert_eq!(TriterDevice::Cuda(1).as_str(), "cuda:1");
        assert_eq!(TriterDevice::Metal.as_str(), "metal");
    }

    #[test]
    fn test_triter_device_default() {
        let device = TriterDevice::default();
        assert_eq!(device, TriterDevice::Cpu);
    }

    #[test]
    fn test_training_config_new() {
        let config = TrainingConfig::new(0.1);
        assert!((config.compression_ratio - 0.1).abs() < f32::EPSILON);
        assert!(config.deterministic);
        assert!(config.seed.is_none());
    }

    #[test]
    fn test_training_config_clamp() {
        let config_low = TrainingConfig::new(0.001);
        assert!((config_low.compression_ratio - 0.01).abs() < f32::EPSILON);

        let config_high = TrainingConfig::new(2.0);
        assert!((config_high.compression_ratio - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_training_config_builder() {
        let config = TrainingConfig::new(0.1)
            .with_deterministic(false)
            .with_seed(42);

        assert!(!config.deterministic);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_inference_config_new() {
        let config = InferenceConfig::new(32);
        assert_eq!(config.batch_size, 32);
        assert!(config.use_ternary_layers);
        assert!(!config.use_kv_cache);
    }

    #[test]
    fn test_inference_config_builder() {
        let config = InferenceConfig::new(16)
            .with_ternary_layers(false)
            .with_kv_cache(2048);

        assert_eq!(config.batch_size, 16);
        assert!(!config.use_ternary_layers);
        assert!(config.use_kv_cache);
        assert_eq!(config.max_seq_len, Some(2048));
    }

    #[test]
    fn test_backend_new() {
        let backend = TriterAccelBackend::new(TriterDevice::Cpu);
        assert_eq!(backend.name(), "tritter-accel");
        assert_eq!(backend.device(), "cpu");
    }

    #[test]
    fn test_backend_cpu() {
        let backend = TriterAccelBackend::cpu();
        assert_eq!(backend.device(), "cpu");
        assert!(backend.is_available());
    }

    #[test]
    fn test_backend_cuda() {
        let backend = TriterAccelBackend::cuda(0);
        assert_eq!(backend.device(), "cuda:0");
    }

    #[test]
    fn test_backend_metal() {
        let backend = TriterAccelBackend::metal();
        assert_eq!(backend.device(), "metal");
    }

    #[test]
    fn test_backend_with_training_mode() {
        let backend = TriterAccelBackend::cpu().with_training_mode(0.1);
        assert!(backend.is_training_mode());
        assert!(!backend.is_inference_mode());

        let config = backend.training_config().unwrap();
        assert!((config.compression_ratio - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_backend_with_inference_mode() {
        let backend = TriterAccelBackend::cpu().with_inference_mode(32);
        assert!(!backend.is_training_mode());
        assert!(backend.is_inference_mode());

        let config = backend.inference_config().unwrap();
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_backend_capabilities_cpu() {
        let backend = TriterAccelBackend::cpu();
        let caps = backend.capabilities();
        assert!(!caps.supports_f16);
        assert!(!caps.supports_bf16);
        assert!(!caps.supports_tf32);
    }

    #[test]
    fn test_backend_capabilities_cuda() {
        let backend = TriterAccelBackend::cuda(0);
        let caps = backend.capabilities();
        assert!(caps.supports_f16);
        assert!(caps.supports_bf16);
        assert!(caps.supports_tf32);
        assert!(caps.max_memory_bytes.is_some());
        assert!(caps.compute_units.is_some());
    }

    #[test]
    fn test_backend_initialize_cpu() {
        let mut backend = TriterAccelBackend::cpu();
        assert!(!backend.is_initialized());

        let result = backend.initialize();
        assert!(result.is_ok());
        assert!(backend.is_initialized());
    }

    #[test]
    fn test_backend_double_initialize() {
        let mut backend = TriterAccelBackend::cpu();
        let _ = backend.initialize();
        let result = backend.initialize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_backend_shutdown() {
        let mut backend = TriterAccelBackend::cpu();
        let _ = backend.initialize();
        assert!(backend.is_initialized());

        let result = backend.shutdown();
        assert!(result.is_ok());
        assert!(!backend.is_initialized());
    }

    #[test]
    fn test_backend_shutdown_not_initialized() {
        let mut backend = TriterAccelBackend::cpu();
        let result = backend.shutdown();
        assert!(result.is_ok());
    }

    #[test]
    fn test_backend_clone() {
        let backend = TriterAccelBackend::cpu().with_training_mode(0.1);
        let cloned = backend.clone();

        assert_eq!(backend.device(), cloned.device());
        assert!(cloned.is_training_mode());
    }

    #[test]
    fn test_backend_default() {
        let backend = TriterAccelBackend::default();
        assert_eq!(backend.device(), "cpu");
        assert!(!backend.is_training_mode());
        assert!(!backend.is_inference_mode());
    }

    #[test]
    fn test_acceleration_mode_default() {
        let mode = AccelerationMode::default();
        assert!(matches!(mode, AccelerationMode::None));
    }

    #[test]
    fn test_backend_device_string_variants() {
        // Test all device string variants
        let backends = vec![
            TriterAccelBackend::cuda(0),
            TriterAccelBackend::cuda(1),
            TriterAccelBackend::cuda(2),
            TriterAccelBackend::cuda(3),
            TriterAccelBackend::cuda(99),
        ];

        let devices: Vec<&str> = backends.iter().map(|b| b.device()).collect();
        assert_eq!(
            devices,
            vec!["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:n"]
        );
    }
}
