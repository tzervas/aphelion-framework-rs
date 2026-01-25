//! Backend abstraction for hardware-specific model execution.
//!
//! This module provides a trait-based abstraction for different computational backends
//! (CPU, GPU, TPU, etc.), enabling hardware-agnostic model definitions that can run
//! on various platforms. It includes device capability reporting and a registry for
//! managing multiple backends.

use crate::diagnostics::TraceSink;
use crate::error::AphelionResult;
use std::collections::HashMap;

/// Device capabilities and features.
///
/// `DeviceCapabilities` describes the hardware features and limits of a computational device,
/// enabling models to make informed decisions about data types and execution strategies.
///
/// # Fields
///
/// * `supports_f16` - Support for 16-bit floating point (half precision)
/// * `supports_bf16` - Support for bfloat16 (brain float)
/// * `supports_tf32` - Support for TensorFlow 32-bit format
/// * `max_memory_bytes` - Maximum available memory in bytes
/// * `compute_units` - Number of parallel compute units (cores, etc.)
///
/// # Examples
///
/// ```
/// use aphelion_core::backend::DeviceCapabilities;
///
/// let caps = DeviceCapabilities {
///     supports_f16: true,
///     supports_bf16: true,
///     supports_tf32: false,
///     max_memory_bytes: Some(8 * 1024 * 1024 * 1024), // 8 GB
///     compute_units: Some(1024),
/// };
/// ```
#[derive(Debug, Clone, Default)]
pub struct DeviceCapabilities {
    /// Device supports 16-bit floating point (FP16)
    pub supports_f16: bool,
    /// Device supports bfloat16 format
    pub supports_bf16: bool,
    /// Device supports TensorFlow 32-bit format
    pub supports_tf32: bool,
    /// Maximum memory available on this device
    pub max_memory_bytes: Option<u64>,
    /// Number of compute units (cores, streaming multiprocessors, etc.)
    pub compute_units: Option<u32>,
}

/// Memory information for a device.
///
/// `MemoryInfo` provides memory usage statistics for a device, useful for monitoring
/// and resource allocation decisions.
///
/// # Fields
///
/// * `total_bytes` - Total memory available
/// * `used_bytes` - Currently used memory
/// * `free_bytes` - Currently available/free memory
///
/// # Examples
///
/// ```
/// use aphelion_core::backend::MemoryInfo;
///
/// let mem = MemoryInfo {
///     total_bytes: 8 * 1024 * 1024 * 1024,
///     used_bytes: 2 * 1024 * 1024 * 1024,
///     free_bytes: 6 * 1024 * 1024 * 1024,
/// };
/// ```
#[derive(Debug, Clone, Default)]
pub struct MemoryInfo {
    /// Total memory in bytes
    pub total_bytes: u64,
    /// Used memory in bytes
    pub used_bytes: u64,
    /// Free/available memory in bytes
    pub free_bytes: u64,
}

/// Backend abstraction for model builds and execution.
///
/// The `Backend` trait defines the interface for hardware-specific implementations.
/// Backends provide information about device capabilities and handle initialization/shutdown.
/// Implementing this trait enables support for different computational targets (CPU, NVIDIA GPU,
/// AMD GPU, TPU, etc.).
///
/// # Implementing Backend
///
/// Types implementing `Backend` must be thread-safe (`Send + Sync`). The trait provides
/// default implementations for `initialize` and `shutdown` that do nothing, suitable for
/// simple backends.
///
/// # Examples
///
/// ```
/// use aphelion_core::backend::{Backend, DeviceCapabilities};
/// use aphelion_core::error::AphelionResult;
///
/// struct CpuBackend;
///
/// impl Backend for CpuBackend {
///     fn name(&self) -> &str { "cpu" }
///     fn device(&self) -> &str { "cpu" }
///     fn capabilities(&self) -> DeviceCapabilities {
///         DeviceCapabilities::default()
///     }
///     fn is_available(&self) -> bool { true }
/// }
/// ```
pub trait Backend: Send + Sync {
    /// Returns the name of this backend (e.g., "cuda", "cpu", "metal")
    fn name(&self) -> &str;

    /// Returns the device identifier (e.g., "cuda:0", "cpu", "gpu_0")
    fn device(&self) -> &str;

    /// Returns the capabilities of the device
    fn capabilities(&self) -> DeviceCapabilities;

    /// Returns whether this backend is currently available for use
    fn is_available(&self) -> bool;

    /// Initializes the backend, preparing it for use.
    ///
    /// This method is called before any operations are performed on the backend.
    /// Default implementation does nothing; override for backends requiring initialization.
    ///
    /// # Errors
    ///
    /// Returns `AphelionError::Backend` if initialization fails.
    fn initialize(&mut self) -> AphelionResult<()> {
        Ok(())
    }

    /// Shuts down the backend and releases resources.
    ///
    /// This method should be called when the backend is no longer needed.
    /// Default implementation does nothing; override for backends requiring cleanup.
    ///
    /// # Errors
    ///
    /// Returns `AphelionError::Backend` if shutdown fails.
    fn shutdown(&mut self) -> AphelionResult<()> {
        Ok(())
    }
}

/// A simple no-op backend for testing and default usage.
///
/// `NullBackend` is a minimal `Backend` implementation that does nothing but report availability.
/// It's useful for testing model architectures without actual hardware dependencies.
///
/// # Examples
///
/// ```
/// use aphelion_core::backend::{Backend, NullBackend};
///
/// let backend = NullBackend::cpu();
/// assert_eq!(backend.name(), "null");
/// assert!(backend.is_available());
/// ```
#[derive(Debug, Clone)]
pub struct NullBackend {
    device: String,
}

impl NullBackend {
    /// Creates a new `NullBackend` with the specified device identifier.
    ///
    /// # Arguments
    ///
    /// * `device` - Device identifier (e.g., "cpu", "gpu")
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::backend::{Backend, NullBackend};
    ///
    /// let backend = NullBackend::new("cpu");
    /// assert_eq!(backend.device(), "cpu");
    /// ```
    pub fn new(device: impl Into<String>) -> Self {
        Self {
            device: device.into(),
        }
    }

    /// Creates a new `NullBackend` configured for CPU execution.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::backend::{Backend, NullBackend};
    ///
    /// let backend = NullBackend::cpu();
    /// assert_eq!(backend.device(), "cpu");
    /// ```
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

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::default()
    }

    fn is_available(&self) -> bool {
        true
    }
}

/// Registry for managing and discovering multiple backends.
///
/// `BackendRegistry` allows registration and retrieval of different `Backend` implementations.
/// This enables applications to dynamically select the appropriate backend at runtime based on
/// availability and requirements.
///
/// # Examples
///
/// ```
/// use aphelion_core::backend::{BackendRegistry, MockBackend};
///
/// let mut registry = BackendRegistry::new();
/// registry.register(Box::new(MockBackend::new("cpu_backend", "cpu")));
/// registry.register(Box::new(MockBackend::new("gpu_backend", "gpu")));
///
/// assert!(registry.get("cpu_backend").is_some());
/// let available = registry.list_available();
/// assert_eq!(available.len(), 2);
/// ```
pub struct BackendRegistry {
    backends: HashMap<String, Box<dyn Backend>>,
}

impl BackendRegistry {
    /// Creates a new, empty backend registry.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::backend::BackendRegistry;
    ///
    /// let registry = BackendRegistry::new();
    /// assert_eq!(registry.list_available().len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
        }
    }

    /// Registers a backend in the registry.
    ///
    /// If a backend with the same name already exists, it will be replaced.
    ///
    /// # Arguments
    ///
    /// * `backend` - The backend implementation to register
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::backend::{BackendRegistry, NullBackend};
    ///
    /// let mut registry = BackendRegistry::new();
    /// registry.register(Box::new(NullBackend::cpu()));
    /// assert_eq!(registry.list_available().len(), 1);
    /// ```
    pub fn register(&mut self, backend: Box<dyn Backend>) {
        let name = backend.name().to_string();
        self.backends.insert(name, backend);
    }

    /// Retrieves a reference to a backend by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The backend name to look up
    ///
    /// # Returns
    ///
    /// `Some(&dyn Backend)` if found, `None` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::backend::{BackendRegistry, NullBackend};
    ///
    /// let mut registry = BackendRegistry::new();
    /// registry.register(Box::new(NullBackend::cpu()));
    ///
    /// assert!(registry.get("null").is_some());
    /// assert!(registry.get("nonexistent").is_none());
    /// ```
    pub fn get(&self, name: &str) -> Option<&dyn Backend> {
        self.backends.get(name).map(|b| b.as_ref())
    }

    /// Lists the names of all available backends.
    ///
    /// # Returns
    ///
    /// A vector of backend names
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::backend::{BackendRegistry, MockBackend};
    ///
    /// let mut registry = BackendRegistry::new();
    /// registry.register(Box::new(MockBackend::new("cpu", "cpu")));
    /// registry.register(Box::new(MockBackend::new("gpu", "gpu")));
    ///
    /// let available = registry.list_available();
    /// assert_eq!(available.len(), 2);
    /// assert!(available.contains(&"cpu"));
    /// ```
    pub fn list_available(&self) -> Vec<&str> {
        self.backends.keys().map(|k| k.as_str()).collect()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock backend implementation for testing purposes.
///
/// `MockBackend` simulates a real backend, allowing tests to verify pipeline behavior
/// without actual hardware. It supports failure injection for both initialization and shutdown.
///
/// # Examples
///
/// ```
/// use aphelion_core::backend::{Backend, MockBackend, DeviceCapabilities};
///
/// let mut backend = MockBackend::new("mock", "test_device");
/// assert!(backend.initialize().is_ok());
/// assert!(backend.init_called());
/// ```
#[derive(Debug, Clone)]
pub struct MockBackend {
    name: String,
    device: String,
    capabilities: DeviceCapabilities,
    is_available: bool,
    should_fail_init: bool,
    should_fail_shutdown: bool,
    init_called: std::sync::Arc<std::sync::atomic::AtomicBool>,
    shutdown_called: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl MockBackend {
    /// Creates a new `MockBackend` with sensible defaults.
    ///
    /// # Arguments
    ///
    /// * `name` - Backend name
    /// * `device` - Device identifier
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::backend::{Backend, MockBackend};
    ///
    /// let backend = MockBackend::new("test", "device");
    /// assert_eq!(backend.name(), "test");
    /// assert_eq!(backend.device(), "device");
    /// assert!(backend.is_available());
    /// ```
    pub fn new(name: impl Into<String>, device: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            device: device.into(),
            capabilities: DeviceCapabilities::default(),
            is_available: true,
            should_fail_init: false,
            should_fail_shutdown: false,
            init_called: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            shutdown_called: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Sets the device capabilities.
    ///
    /// # Arguments
    ///
    /// * `capabilities` - The device capabilities to report
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::backend::{MockBackend, DeviceCapabilities};
    ///
    /// let caps = DeviceCapabilities {
    ///     supports_f16: true,
    ///     ..Default::default()
    /// };
    /// let backend = MockBackend::new("test", "gpu")
    ///     .with_capabilities(caps);
    /// ```
    pub fn with_capabilities(mut self, capabilities: DeviceCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Sets whether the backend reports as available.
    ///
    /// # Arguments
    ///
    /// * `available` - Availability status
    pub fn with_availability(mut self, available: bool) -> Self {
        self.is_available = available;
        self
    }

    /// Configures the backend to fail during initialization.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::backend::{Backend, MockBackend};
    ///
    /// let mut backend = MockBackend::new("test", "device")
    ///     .with_init_failure();
    /// assert!(backend.initialize().is_err());
    /// ```
    pub fn with_init_failure(mut self) -> Self {
        self.should_fail_init = true;
        self
    }

    /// Configures the backend to fail during shutdown.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::backend::{Backend, MockBackend};
    ///
    /// let mut backend = MockBackend::new("test", "device")
    ///     .with_shutdown_failure();
    /// assert!(backend.shutdown().is_err());
    /// ```
    pub fn with_shutdown_failure(mut self) -> Self {
        self.should_fail_shutdown = true;
        self
    }

    /// Returns whether `initialize` was called on this backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::backend::{Backend, MockBackend};
    ///
    /// let mut backend = MockBackend::new("test", "device");
    /// assert!(!backend.init_called());
    /// let _ = backend.initialize();
    /// assert!(backend.init_called());
    /// ```
    pub fn init_called(&self) -> bool {
        self.init_called.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Returns whether `shutdown` was called on this backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::backend::{Backend, MockBackend};
    ///
    /// let mut backend = MockBackend::new("test", "device");
    /// assert!(!backend.shutdown_called());
    /// let _ = backend.shutdown();
    /// assert!(backend.shutdown_called());
    /// ```
    pub fn shutdown_called(&self) -> bool {
        self.shutdown_called.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl Backend for MockBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn device(&self) -> &str {
        &self.device
    }

    fn capabilities(&self) -> DeviceCapabilities {
        self.capabilities.clone()
    }

    fn is_available(&self) -> bool {
        self.is_available
    }

    fn initialize(&mut self) -> AphelionResult<()> {
        self.init_called
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if self.should_fail_init {
            Err(crate::error::AphelionError::backend(
                "MockBackend initialization failed",
            ))
        } else {
            Ok(())
        }
    }

    fn shutdown(&mut self) -> AphelionResult<()> {
        self.shutdown_called
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if self.should_fail_shutdown {
            Err(crate::error::AphelionError::backend(
                "MockBackend shutdown failed",
            ))
        } else {
            Ok(())
        }
    }
}

/// Trait for building models with backend and tracing support.
///
/// `ModelBuilder` defines a common interface for constructing models. Implementations
/// receive both a backend and a trace sink, enabling hardware-specific optimizations
/// and comprehensive logging of the build process.
///
/// # Associated Types
///
/// * `Output` - The type produced by the build operation
///
/// # Examples
///
/// ```
/// use aphelion_core::backend::{ModelBuilder, Backend};
/// use aphelion_core::diagnostics::TraceSink;
/// use aphelion_core::graph::BuildGraph;
///
/// struct SimpleBuilder;
///
/// impl ModelBuilder for SimpleBuilder {
///     type Output = BuildGraph;
///
///     fn build(&self, _backend: &dyn Backend, _trace: &dyn TraceSink) -> BuildGraph {
///         BuildGraph::default()
///     }
/// }
/// ```
pub trait ModelBuilder: Send + Sync {
    /// The type produced by the build operation
    type Output;

    /// Builds a model using the given backend and trace sink.
    ///
    /// # Arguments
    ///
    /// * `backend` - The computational backend to use
    /// * `trace` - The trace sink for recording build events
    fn build(&self, backend: &dyn Backend, trace: &dyn TraceSink) -> Self::Output;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_capabilities_default() {
        let caps = DeviceCapabilities::default();
        assert!(!caps.supports_f16);
        assert!(!caps.supports_bf16);
        assert!(!caps.supports_tf32);
        assert_eq!(caps.max_memory_bytes, None);
        assert_eq!(caps.compute_units, None);
    }

    #[test]
    fn test_device_capabilities_custom() {
        let caps = DeviceCapabilities {
            supports_f16: true,
            supports_bf16: true,
            supports_tf32: false,
            max_memory_bytes: Some(1024 * 1024 * 1024),
            compute_units: Some(512),
        };
        assert!(caps.supports_f16);
        assert!(caps.supports_bf16);
        assert!(!caps.supports_tf32);
        assert_eq!(caps.max_memory_bytes, Some(1024 * 1024 * 1024));
        assert_eq!(caps.compute_units, Some(512));
    }

    #[test]
    fn test_device_capabilities_clone() {
        let caps1 = DeviceCapabilities {
            supports_f16: true,
            supports_bf16: false,
            supports_tf32: true,
            max_memory_bytes: Some(2048),
            compute_units: Some(256),
        };
        let caps2 = caps1.clone();
        assert_eq!(caps1.supports_f16, caps2.supports_f16);
        assert_eq!(caps1.supports_bf16, caps2.supports_bf16);
        assert_eq!(caps1.max_memory_bytes, caps2.max_memory_bytes);
    }

    #[test]
    fn test_memory_info_default() {
        let mem = MemoryInfo::default();
        assert_eq!(mem.total_bytes, 0);
        assert_eq!(mem.used_bytes, 0);
        assert_eq!(mem.free_bytes, 0);
    }

    #[test]
    fn test_null_backend_name() {
        let backend = NullBackend::cpu();
        assert_eq!(backend.name(), "null");
    }

    #[test]
    fn test_null_backend_device() {
        let backend = NullBackend::cpu();
        assert_eq!(backend.device(), "cpu");
    }

    #[test]
    fn test_null_backend_capabilities() {
        let backend = NullBackend::cpu();
        let caps = backend.capabilities();
        assert!(!caps.supports_f16);
        assert!(!caps.supports_bf16);
        assert!(!caps.supports_tf32);
    }

    #[test]
    fn test_null_backend_is_available() {
        let backend = NullBackend::cpu();
        assert!(backend.is_available());
    }

    #[test]
    fn test_null_backend_initialize() {
        let mut backend = NullBackend::cpu();
        let result = backend.initialize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_null_backend_shutdown() {
        let mut backend = NullBackend::cpu();
        let result = backend.shutdown();
        assert!(result.is_ok());
    }

    #[test]
    fn test_backend_registry_new() {
        let registry = BackendRegistry::new();
        assert_eq!(registry.list_available().len(), 0);
    }

    #[test]
    fn test_backend_registry_register() {
        let mut registry = BackendRegistry::new();
        let backend = Box::new(NullBackend::cpu());
        registry.register(backend);
        assert_eq!(registry.list_available().len(), 1);
    }

    #[test]
    fn test_backend_registry_get() {
        let mut registry = BackendRegistry::new();
        let backend = Box::new(NullBackend::cpu());
        registry.register(backend);
        let retrieved = registry.get("null");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name(), "null");
    }

    #[test]
    fn test_backend_registry_get_missing() {
        let registry = BackendRegistry::new();
        let retrieved = registry.get("nonexistent");
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_backend_registry_list_available() {
        let mut registry = BackendRegistry::new();
        registry.register(Box::new(NullBackend::new("cpu")));
        registry.register(Box::new(MockBackend::new("mock", "gpu")));

        let available = registry.list_available();
        assert_eq!(available.len(), 2);
        assert!(available.contains(&"null"));
        assert!(available.contains(&"mock"));
    }

    #[test]
    fn test_backend_registry_default() {
        let registry = BackendRegistry::default();
        assert_eq!(registry.list_available().len(), 0);
    }

    #[test]
    fn test_mock_backend_creation() {
        let backend = MockBackend::new("mock", "cpu");
        assert_eq!(backend.name(), "mock");
        assert_eq!(backend.device(), "cpu");
        assert!(backend.is_available());
    }

    #[test]
    fn test_mock_backend_with_capabilities() {
        let caps = DeviceCapabilities {
            supports_f16: true,
            supports_bf16: true,
            supports_tf32: false,
            max_memory_bytes: Some(8 * 1024 * 1024 * 1024),
            compute_units: Some(1024),
        };
        let backend = MockBackend::new("mock", "gpu").with_capabilities(caps.clone());
        let retrieved_caps = backend.capabilities();
        assert!(retrieved_caps.supports_f16);
        assert!(retrieved_caps.supports_bf16);
        assert!(!retrieved_caps.supports_tf32);
        assert_eq!(retrieved_caps.max_memory_bytes, Some(8 * 1024 * 1024 * 1024));
        assert_eq!(retrieved_caps.compute_units, Some(1024));
    }

    #[test]
    fn test_mock_backend_with_availability() {
        let backend = MockBackend::new("mock", "cpu").with_availability(false);
        assert!(!backend.is_available());
    }

    #[test]
    fn test_mock_backend_initialize_success() {
        let mut backend = MockBackend::new("mock", "cpu");
        let result = backend.initialize();
        assert!(result.is_ok());
        assert!(backend.init_called());
    }

    #[test]
    fn test_mock_backend_initialize_failure() {
        let mut backend = MockBackend::new("mock", "cpu").with_init_failure();
        let result = backend.initialize();
        assert!(result.is_err());
        assert!(backend.init_called());
    }

    #[test]
    fn test_mock_backend_shutdown_success() {
        let mut backend = MockBackend::new("mock", "cpu");
        let result = backend.shutdown();
        assert!(result.is_ok());
        assert!(backend.shutdown_called());
    }

    #[test]
    fn test_mock_backend_shutdown_failure() {
        let mut backend = MockBackend::new("mock", "cpu").with_shutdown_failure();
        let result = backend.shutdown();
        assert!(result.is_err());
        assert!(backend.shutdown_called());
    }

    #[test]
    fn test_mock_backend_lifecycle() {
        let mut backend = MockBackend::new("mock", "cpu");
        assert!(!backend.init_called());
        assert!(!backend.shutdown_called());

        let init_result = backend.initialize();
        assert!(init_result.is_ok());
        assert!(backend.init_called());
        assert!(!backend.shutdown_called());

        let shutdown_result = backend.shutdown();
        assert!(shutdown_result.is_ok());
        assert!(backend.init_called());
        assert!(backend.shutdown_called());
    }

    #[test]
    fn test_mock_backend_builder_pattern() {
        let caps = DeviceCapabilities {
            supports_f16: true,
            supports_bf16: false,
            supports_tf32: true,
            max_memory_bytes: Some(1024),
            compute_units: Some(128),
        };
        let backend = MockBackend::new("test_backend", "gpu")
            .with_capabilities(caps)
            .with_availability(true);

        assert_eq!(backend.name(), "test_backend");
        assert_eq!(backend.device(), "gpu");
        assert!(backend.is_available());
        let retrieved_caps = backend.capabilities();
        assert!(retrieved_caps.supports_f16);
        assert!(!retrieved_caps.supports_bf16);
    }

    #[test]
    fn test_mock_backend_clone() {
        let backend1 = MockBackend::new("mock", "cpu").with_availability(false);
        let backend2 = backend1.clone();
        assert_eq!(backend1.name(), backend2.name());
        assert_eq!(backend1.device(), backend2.device());
        assert_eq!(backend1.is_available(), backend2.is_available());
    }
}
