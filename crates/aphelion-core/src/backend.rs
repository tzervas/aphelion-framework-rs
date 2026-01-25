use crate::diagnostics::TraceSink;
use crate::error::AphelionResult;
use std::collections::HashMap;

/// Device capabilities and features.
#[derive(Debug, Clone, Default)]
pub struct DeviceCapabilities {
    pub supports_f16: bool,
    pub supports_bf16: bool,
    pub supports_tf32: bool,
    pub max_memory_bytes: Option<u64>,
    pub compute_units: Option<u32>,
}

/// Memory information for a device.
#[derive(Debug, Clone, Default)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub free_bytes: u64,
}

/// Backend abstraction for model builds and execution.
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn device(&self) -> &str;
    fn capabilities(&self) -> DeviceCapabilities;
    fn is_available(&self) -> bool;
    fn initialize(&mut self) -> AphelionResult<()> {
        Ok(())
    }
    fn shutdown(&mut self) -> AphelionResult<()> {
        Ok(())
    }
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

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::default()
    }

    fn is_available(&self) -> bool {
        true
    }
}

/// Registry for managing multiple backends.
pub struct BackendRegistry {
    backends: HashMap<String, Box<dyn Backend>>,
}

impl BackendRegistry {
    /// Create a new, empty backend registry.
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
        }
    }

    /// Register a backend in the registry.
    pub fn register(&mut self, backend: Box<dyn Backend>) {
        let name = backend.name().to_string();
        self.backends.insert(name, backend);
    }

    /// Get a reference to a backend by name.
    pub fn get(&self, name: &str) -> Option<&dyn Backend> {
        self.backends.get(name).map(|b| b.as_ref())
    }

    /// List all available backends.
    pub fn list_available(&self) -> Vec<&str> {
        self.backends.keys().map(|k| k.as_str()).collect()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock backend for testing purposes.
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
    /// Create a new MockBackend with defaults.
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

    /// Configure capabilities.
    pub fn with_capabilities(mut self, capabilities: DeviceCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    /// Set availability.
    pub fn with_availability(mut self, available: bool) -> Self {
        self.is_available = available;
        self
    }

    /// Configure to fail on initialize.
    pub fn with_init_failure(mut self) -> Self {
        self.should_fail_init = true;
        self
    }

    /// Configure to fail on shutdown.
    pub fn with_shutdown_failure(mut self) -> Self {
        self.should_fail_shutdown = true;
        self
    }

    /// Check if initialize was called.
    pub fn init_called(&self) -> bool {
        self.init_called.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Check if shutdown was called.
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
            Err(crate::error::AphelionError::Backend(
                "MockBackend initialization failed".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    fn shutdown(&mut self) -> AphelionResult<()> {
        self.shutdown_called
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if self.should_fail_shutdown {
            Err(crate::error::AphelionError::Backend(
                "MockBackend shutdown failed".to_string(),
            ))
        } else {
            Ok(())
        }
    }
}

/// Simple model builder interface.
pub trait ModelBuilder: Send + Sync {
    type Output;

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
