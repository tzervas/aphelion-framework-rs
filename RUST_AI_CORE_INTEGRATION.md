# Task: Integrate rust-ai-core into aphelion-framework-rs

## Objective
Fully integrate the `rust-ai-core` crate (https://github.com/tzervas/rust-ai-core, v0.2.0) into aphelion-framework-rs, replacing placeholder types with real implementations. Access all tensor operations through rust-ai-core's interop layer. Include MemoryTracker in BuildContext and expose CubeCL kernel utilities.

## Context
- **aphelion-framework-rs**: AI model pipeline framework at `/home/kang/Documents/projects/github/aphelion-framework-rs`
- **rust-ai-core**: Foundation crate providing device management, memory tracking, dtype utilities, CubeCL↔Candle interop, and GPU dispatch traits
- **Current state**: `crates/aphelion-core/src/rust_ai_core.rs` has ~1000 lines of placeholder types (RacDevice, RacModelConfig, etc.)

## rust-ai-core Public API (v0.2.0)

```rust
// Device management
pub use device::{get_device, warn_if_cpu, DeviceConfig};

// Memory tracking
pub use memory::{estimate_tensor_bytes, MemoryTracker, estimate_attention_memory, DEFAULT_OVERHEAD_FACTOR};

// DType utilities
pub use dtype::{bytes_per_element, is_floating_point, DTypeExt, PrecisionMode};

// Error types
pub use error::{CoreError, Result};

// Traits
pub use traits::{ValidatableConfig, Quantize, Dequantize, GpuDispatchable};

// CubeCL interop (requires "cuda" feature)
#[cfg(feature = "cuda")]
pub use cubecl::{
    TensorBuffer, candle_to_cubecl_handle, cubecl_to_candle_tensor,
    allocate_output_buffer, has_cubecl_cuda_support,
};

// Logging
pub use logging::{init_logging, LogConfig};

pub const VERSION: &str = "0.2.0";
```

## Implementation Steps

### Step 1: Update Cargo.toml Dependencies

**File: `crates/aphelion-core/Cargo.toml`**

Add rust-ai-core as optional dependency:
```toml
[dependencies]
# ... existing deps ...
rust-ai-core = { version = "0.2", optional = true }
candle-core = { version = "0.8", optional = true }  # Re-export Device type

[features]
# ... existing features ...
rust-ai-core = ["dep:rust-ai-core", "dep:candle-core"]
cuda = ["rust-ai-core/cuda"]  # Pass through CUDA feature
```

**File: `crates/aphelion-python/Cargo.toml`**

Add feature passthrough:
```toml
[features]
rust-ai-core = ["aphelion-core/rust-ai-core"]
cuda = ["aphelion-core/cuda"]
```

### Step 2: Refactor rust_ai_core.rs

**File: `crates/aphelion-core/src/rust_ai_core.rs`**

Replace placeholder types with conditional imports. Keep placeholders for when feature is disabled.

```rust
//! rust-ai-core integration module.
//!
//! When the `rust-ai-core` feature is enabled, this module re-exports
//! actual types from the rust-ai-core crate. Otherwise, it provides
//! placeholder types for API compatibility.

use crate::config::ModelConfig;
use crate::error::{AphelionError, AphelionResult};
use crate::graph::{BuildGraph, NodeId};

// ============================================================================
// Conditional Imports
// ============================================================================

#[cfg(feature = "rust-ai-core")]
pub mod real {
    //! Real rust-ai-core types (feature enabled)
    
    pub use rust_ai_core::{
        // Device
        DeviceConfig, get_device, warn_if_cpu,
        // Memory
        MemoryTracker, estimate_tensor_bytes,
        memory::{estimate_attention_memory, DEFAULT_OVERHEAD_FACTOR},
        // DType
        DTypeExt, bytes_per_element, is_floating_point,
        dtype::PrecisionMode,
        // Error
        CoreError, Result as RacResult,
        // Traits
        ValidatableConfig, Quantize, Dequantize, GpuDispatchable,
        // Logging
        init_logging, LogConfig,
        // Version
        VERSION as RAC_VERSION,
    };
    
    #[cfg(feature = "cuda")]
    pub use rust_ai_core::{
        TensorBuffer, candle_to_cubecl_handle, cubecl_to_candle_tensor,
        allocate_output_buffer, has_cubecl_cuda_support,
    };
    
    pub use candle_core::{Device, DType, Tensor};
}

#[cfg(not(feature = "rust-ai-core"))]
pub mod placeholder {
    //! Placeholder types (feature disabled)
    // Keep existing placeholder implementations...
}

// Re-export based on feature
#[cfg(feature = "rust-ai-core")]
pub use real::*;

#[cfg(not(feature = "rust-ai-core"))]
pub use placeholder::*;
```

### Step 3: Create Device Bridge

**File: `crates/aphelion-core/src/rust_ai_core.rs`** (add to module)

```rust
#[cfg(feature = "rust-ai-core")]
mod device_bridge {
    use super::*;
    use candle_core::Device;
    
    /// Bridge between aphelion's RacDevice and candle's Device
    #[derive(Debug, Clone)]
    pub struct AphelionDevice {
        inner: Device,
        config: DeviceConfig,
    }
    
    impl AphelionDevice {
        /// Create from DeviceConfig, auto-selecting best available device
        pub fn from_config(config: DeviceConfig) -> AphelionResult<Self> {
            let device = get_device(&config)
                .map_err(|e| AphelionError::backend(format!("Device selection failed: {}", e)))?;
            Ok(Self { inner: device, config })
        }
        
        /// Create CPU device
        pub fn cpu() -> Self {
            Self {
                inner: Device::Cpu,
                config: DeviceConfig::new().with_force_cpu(true),
            }
        }
        
        /// Create CUDA device
        pub fn cuda(ordinal: usize) -> AphelionResult<Self> {
            let config = DeviceConfig::new().with_cuda_device(ordinal);
            Self::from_config(config)
        }
        
        /// Get the underlying candle Device
        pub fn as_candle_device(&self) -> &Device {
            &self.inner
        }
        
        /// Check if this is a CUDA device
        pub fn is_cuda(&self) -> bool {
            matches!(self.inner, Device::Cuda(_))
        }
        
        /// Warn if running on CPU (for performance-critical paths)
        pub fn warn_if_cpu(&self, crate_name: &str) {
            warn_if_cpu(&self.inner, crate_name);
        }
    }
    
    impl From<Device> for AphelionDevice {
        fn from(device: Device) -> Self {
            let config = match &device {
                Device::Cpu => DeviceConfig::new().with_force_cpu(true),
                Device::Cuda(cuda) => DeviceConfig::new().with_cuda_device(cuda.ordinal()),
                _ => DeviceConfig::default(),
            };
            Self { inner: device, config }
        }
    }
}

#[cfg(feature = "rust-ai-core")]
pub use device_bridge::AphelionDevice;
```

### Step 4: Add MemoryTracker to BuildContext

**File: `crates/aphelion-core/src/pipeline.rs`**

Update BuildContext to include optional MemoryTracker:

```rust
#[cfg(feature = "rust-ai-core")]
use crate::rust_ai_core::{MemoryTracker, estimate_tensor_bytes};

/// Execution context for build pipelines.
pub struct BuildContext<'a> {
    pub backend: &'a dyn Backend,
    pub trace: &'a dyn TraceSink,
    
    #[cfg(feature = "rust-ai-core")]
    pub memory_tracker: Option<&'a MemoryTracker>,
}

impl<'a> BuildContext<'a> {
    /// Create context with memory tracking
    #[cfg(feature = "rust-ai-core")]
    pub fn with_memory_tracker(
        backend: &'a dyn Backend,
        trace: &'a dyn TraceSink,
        memory_tracker: &'a MemoryTracker,
    ) -> Self {
        Self {
            backend,
            trace,
            memory_tracker: Some(memory_tracker),
        }
    }
    
    /// Check if allocation would fit in memory budget
    #[cfg(feature = "rust-ai-core")]
    pub fn would_fit(&self, bytes: usize) -> bool {
        self.memory_tracker
            .map(|t| t.would_fit(bytes))
            .unwrap_or(true)
    }
    
    /// Record an allocation
    #[cfg(feature = "rust-ai-core")]
    pub fn allocate(&self, bytes: usize) -> AphelionResult<()> {
        if let Some(tracker) = self.memory_tracker {
            tracker.allocate(bytes)
                .map_err(|e| AphelionError::backend(format!("OOM: {}", e)))?;
        }
        Ok(())
    }
    
    /// Record a deallocation
    #[cfg(feature = "rust-ai-core")]
    pub fn deallocate(&self, bytes: usize) {
        if let Some(tracker) = self.memory_tracker {
            tracker.deallocate(bytes);
        }
    }
}
```

### Step 5: Expose CubeCL Interop

**File: `crates/aphelion-core/src/rust_ai_core.rs`** (add module)

```rust
#[cfg(all(feature = "rust-ai-core", feature = "cuda"))]
pub mod cubecl {
    //! CubeCL ↔ Candle tensor interoperability.
    //!
    //! Provides utilities for converting between Candle tensors and CubeCL
    //! buffer handles for custom GPU kernel execution.
    
    pub use rust_ai_core::{
        TensorBuffer,
        candle_to_cubecl_handle,
        cubecl_to_candle_tensor,
        allocate_output_buffer,
        has_cubecl_cuda_support,
    };
    
    use crate::error::{AphelionError, AphelionResult};
    use candle_core::{Device, DType, Tensor};
    
    /// Wrapper for CubeCL operations with error translation
    pub struct CubeclContext {
        device: Device,
    }
    
    impl CubeclContext {
        /// Create a CubeCL context for the given device
        pub fn new(device: Device) -> AphelionResult<Self> {
            if !matches!(device, Device::Cuda(_)) {
                return Err(AphelionError::backend(
                    "CubeCL requires CUDA device"
                ));
            }
            if !has_cubecl_cuda_support() {
                return Err(AphelionError::backend(
                    "CubeCL CUDA support not available"
                ));
            }
            Ok(Self { device })
        }
        
        /// Convert Candle tensor to CubeCL buffer for kernel input
        pub fn tensor_to_buffer(&self, tensor: &Tensor) -> AphelionResult<TensorBuffer> {
            candle_to_cubecl_handle(tensor)
                .map_err(|e| AphelionError::backend(format!("CubeCL conversion failed: {}", e)))
        }
        
        /// Convert CubeCL buffer back to Candle tensor after kernel execution
        pub fn buffer_to_tensor(&self, buffer: &TensorBuffer) -> AphelionResult<Tensor> {
            cubecl_to_candle_tensor(buffer, &self.device)
                .map_err(|e| AphelionError::backend(format!("CubeCL conversion failed: {}", e)))
        }
        
        /// Allocate output buffer for kernel results
        pub fn alloc_output(&self, shape: &[usize], dtype: DType) -> AphelionResult<TensorBuffer> {
            allocate_output_buffer(shape, dtype)
                .map_err(|e| AphelionError::backend(format!("CubeCL allocation failed: {}", e)))
        }
    }
}

#[cfg(all(feature = "rust-ai-core", feature = "cuda"))]
pub use cubecl::CubeclContext;
```

### Step 6: Add GpuDispatchable Pipeline Stage

**File: `crates/aphelion-core/src/pipeline.rs`** (add)

```rust
#[cfg(feature = "rust-ai-core")]
use crate::rust_ai_core::GpuDispatchable;

/// A pipeline stage that uses GpuDispatchable for automatic GPU/CPU routing
#[cfg(feature = "rust-ai-core")]
pub struct GpuDispatchStage<Op: GpuDispatchable> {
    name: String,
    operation: Op,
}

#[cfg(feature = "rust-ai-core")]
impl<Op: GpuDispatchable> GpuDispatchStage<Op> {
    pub fn new(name: impl Into<String>, operation: Op) -> Self {
        Self {
            name: name.into(),
            operation,
        }
    }
}

#[cfg(feature = "rust-ai-core")]
impl<Op: GpuDispatchable<Input = BuildGraph, Output = BuildGraph>> PipelineStage 
    for GpuDispatchStage<Op> 
{
    fn name(&self) -> &str {
        &self.name
    }
    
    fn execute(&self, ctx: &BuildContext<'_>, graph: &mut BuildGraph) -> AphelionResult<()> {
        use crate::rust_ai_core::AphelionDevice;
        
        // Get device from backend
        let device = AphelionDevice::cpu(); // TODO: Get from backend
        
        // Dispatch to GPU or CPU automatically
        let result = self.operation
            .dispatch(graph, device.as_candle_device())
            .map_err(|e| AphelionError::pipeline(format!("GPU dispatch failed: {}", e)))?;
        
        *graph = result;
        Ok(())
    }
}
```

### Step 7: Update Python Bindings

**File: `crates/aphelion-python/src/lib.rs`**

Add rust-ai-core Python bindings:

```rust
#[cfg(feature = "rust-ai-core")]
mod rust_ai_core_bindings;

#[pymodule]
fn aphelion(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ... existing registrations ...
    
    // rust-ai-core feature flag
    m.add("HAS_RUST_AI_CORE", cfg!(feature = "rust-ai-core"))?;
    
    #[cfg(feature = "rust-ai-core")]
    rust_ai_core_bindings::register(m)?;
    
    Ok(())
}
```

**File: `crates/aphelion-python/src/rust_ai_core_bindings.rs`** (new file)

```rust
//! Python bindings for rust-ai-core types.

use pyo3::prelude::*;
use aphelion_core::rust_ai_core::{
    DeviceConfig, get_device, MemoryTracker, estimate_tensor_bytes,
    DEFAULT_OVERHEAD_FACTOR, AphelionDevice,
};

#[pyclass(name = "DeviceConfig")]
#[derive(Clone)]
pub struct PyDeviceConfig {
    inner: DeviceConfig,
}

#[pymethods]
impl PyDeviceConfig {
    #[new]
    fn new() -> Self {
        Self { inner: DeviceConfig::new() }
    }
    
    fn with_cuda_device(&self, ordinal: usize) -> Self {
        Self { inner: self.inner.clone().with_cuda_device(ordinal) }
    }
    
    fn with_force_cpu(&self, force: bool) -> Self {
        Self { inner: self.inner.clone().with_force_cpu(force) }
    }
    
    fn with_crate_name(&self, name: &str) -> Self {
        Self { inner: self.inner.clone().with_crate_name(name) }
    }
}

#[pyclass(name = "Device")]
pub struct PyDevice {
    inner: AphelionDevice,
}

#[pymethods]
impl PyDevice {
    #[staticmethod]
    fn cpu() -> Self {
        Self { inner: AphelionDevice::cpu() }
    }
    
    #[staticmethod]
    fn cuda(ordinal: usize) -> PyResult<Self> {
        AphelionDevice::cuda(ordinal)
            .map(|d| Self { inner: d })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    #[staticmethod]
    fn from_config(config: &PyDeviceConfig) -> PyResult<Self> {
        AphelionDevice::from_config(config.inner.clone())
            .map(|d| Self { inner: d })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
    
    fn is_cuda(&self) -> bool {
        self.inner.is_cuda()
    }
    
    fn __repr__(&self) -> String {
        format!("Device(cuda={})", self.inner.is_cuda())
    }
}

#[pyclass(name = "MemoryTracker")]
pub struct PyMemoryTracker {
    inner: MemoryTracker,
}

#[pymethods]
impl PyMemoryTracker {
    #[new]
    fn new() -> Self {
        Self { inner: MemoryTracker::new() }
    }
    
    #[staticmethod]
    fn with_limit(limit_bytes: usize) -> Self {
        Self { inner: MemoryTracker::with_limit(limit_bytes) }
    }
    
    fn with_overhead_factor(&self, factor: f64) -> Self {
        Self { inner: self.inner.clone().with_overhead_factor(factor) }
    }
    
    fn allocate(&self, bytes: usize) -> PyResult<()> {
        self.inner.allocate(bytes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyMemoryError, _>(e.to_string()))
    }
    
    fn deallocate(&self, bytes: usize) {
        self.inner.deallocate(bytes);
    }
    
    fn would_fit(&self, bytes: usize) -> bool {
        self.inner.would_fit(bytes)
    }
    
    fn allocated_bytes(&self) -> usize {
        self.inner.allocated_bytes()
    }
    
    fn peak_bytes(&self) -> usize {
        self.inner.peak_bytes()
    }
    
    fn limit_bytes(&self) -> usize {
        self.inner.limit_bytes()
    }
    
    fn estimate_with_overhead(&self, shape: Vec<usize>, dtype: &str) -> PyResult<usize> {
        let dt = parse_dtype(dtype)?;
        Ok(self.inner.estimate_with_overhead(&shape, dt))
    }
}

#[pyfunction]
fn estimate_memory(shape: Vec<usize>, dtype: &str) -> PyResult<usize> {
    let dt = parse_dtype(dtype)?;
    Ok(estimate_tensor_bytes(&shape, dt))
}

#[pyfunction]
fn get_default_overhead_factor() -> f64 {
    DEFAULT_OVERHEAD_FACTOR
}

// CubeCL bindings (when cuda feature enabled)
#[cfg(feature = "cuda")]
mod cubecl_bindings {
    use super::*;
    use aphelion_core::rust_ai_core::cubecl::*;
    
    #[pyfunction]
    pub fn has_cuda_support() -> bool {
        has_cubecl_cuda_support()
    }
}

fn parse_dtype(s: &str) -> PyResult<candle_core::DType> {
    use candle_core::DType;
    match s.to_lowercase().as_str() {
        "f32" | "float32" => Ok(DType::F32),
        "f16" | "float16" => Ok(DType::F16),
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f64" | "float64" => Ok(DType::F64),
        "i32" | "int32" => Ok(DType::I32),
        "i64" | "int64" => Ok(DType::I64),
        "u8" | "uint8" => Ok(DType::U8),
        "u32" | "uint32" => Ok(DType::U32),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown dtype: {}", s)
        )),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDeviceConfig>()?;
    m.add_class::<PyDevice>()?;
    m.add_class::<PyMemoryTracker>()?;
    m.add_function(wrap_pyfunction!(estimate_memory, m)?)?;
    m.add_function(wrap_pyfunction!(get_default_overhead_factor, m)?)?;
    
    #[cfg(feature = "cuda")]
    {
        m.add("HAS_CUDA", true)?;
        m.add_function(wrap_pyfunction!(cubecl_bindings::has_cuda_support, m)?)?;
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        m.add("HAS_CUDA", false)?;
    }
    
    Ok(())
}
```

### Step 8: Update Python Type Stubs

**File: `crates/aphelion-python/python/aphelion/__init__.pyi`**

Add type stubs for new bindings:

```python
# rust-ai-core types (available when HAS_RUST_AI_CORE is True)

class DeviceConfig:
    """Configuration for device selection."""
    def __init__(self) -> None: ...
    def with_cuda_device(self, ordinal: int) -> DeviceConfig: ...
    def with_force_cpu(self, force: bool) -> DeviceConfig: ...
    def with_crate_name(self, name: str) -> DeviceConfig: ...

class Device:
    """Compute device (CPU or CUDA)."""
    @staticmethod
    def cpu() -> Device: ...
    @staticmethod
    def cuda(ordinal: int) -> Device: ...
    @staticmethod
    def from_config(config: DeviceConfig) -> Device: ...
    def is_cuda(self) -> bool: ...

class MemoryTracker:
    """GPU memory tracking and budget management."""
    def __init__(self) -> None: ...
    @staticmethod
    def with_limit(limit_bytes: int) -> MemoryTracker: ...
    def with_overhead_factor(self, factor: float) -> MemoryTracker: ...
    def allocate(self, bytes: int) -> None: ...
    def deallocate(self, bytes: int) -> None: ...
    def would_fit(self, bytes: int) -> bool: ...
    def allocated_bytes(self) -> int: ...
    def peak_bytes(self) -> int: ...
    def limit_bytes(self) -> int: ...
    def estimate_with_overhead(self, shape: list[int], dtype: str) -> int: ...

def estimate_memory(shape: list[int], dtype: str) -> int:
    """Estimate memory in bytes for a tensor with given shape and dtype."""
    ...

def get_default_overhead_factor() -> float:
    """Get the default memory overhead factor (1.1x)."""
    ...

# Feature flags
HAS_RUST_AI_CORE: bool
HAS_CUDA: bool

def has_cuda_support() -> bool:
    """Check if CubeCL CUDA support is available at runtime."""
    ...
```

### Step 9: Update lib.rs Exports

**File: `crates/aphelion-core/src/lib.rs`**

Ensure proper re-exports:

```rust
pub mod rust_ai_core;

#[cfg(feature = "rust-ai-core")]
pub use rust_ai_core::{
    // Device
    AphelionDevice, DeviceConfig, get_device, warn_if_cpu,
    // Memory  
    MemoryTracker, estimate_tensor_bytes,
    // DType
    DTypeExt, bytes_per_element, is_floating_point, PrecisionMode,
    // Traits
    ValidatableConfig, Quantize, Dequantize, GpuDispatchable,
    // Candle types
    Device, DType, Tensor,
};

#[cfg(all(feature = "rust-ai-core", feature = "cuda"))]
pub use rust_ai_core::cubecl::{
    CubeclContext, TensorBuffer, 
    candle_to_cubecl_handle, cubecl_to_candle_tensor,
    allocate_output_buffer, has_cubecl_cuda_support,
};
```

### Step 10: Add Integration Tests

**File: `crates/aphelion-tests/src/lib.rs`** (add test module)

```rust
#[cfg(all(test, feature = "rust-ai-core"))]
mod rust_ai_core_integration {
    use aphelion_core::rust_ai_core::*;
    use aphelion_core::pipeline::BuildContext;
    use aphelion_core::backend::NullBackend;
    use aphelion_core::diagnostics::InMemoryTraceSink;
    
    #[test]
    fn test_device_selection() {
        let config = DeviceConfig::new().with_force_cpu(true);
        let device = AphelionDevice::from_config(config).unwrap();
        assert!(!device.is_cuda());
    }
    
    #[test]
    fn test_memory_tracker_in_context() {
        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let tracker = MemoryTracker::with_limit(1024 * 1024 * 1024); // 1GB
        
        let ctx = BuildContext::with_memory_tracker(&backend, &trace, &tracker);
        
        // Estimate memory for a tensor
        let bytes = estimate_tensor_bytes(&[32, 64, 128], candle_core::DType::F32);
        assert!(ctx.would_fit(bytes));
        
        ctx.allocate(bytes).unwrap();
        assert_eq!(tracker.allocated_bytes(), bytes);
    }
    
    #[test]
    fn test_dtype_utilities() {
        use candle_core::DType;
        
        assert!(DType::F32.is_training_dtype());
        assert!(DType::BF16.is_half_precision());
        assert_eq!(DType::F16.accumulator_dtype(), DType::F32);
        assert_eq!(bytes_per_element(DType::F32), 4);
    }
    
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "Requires CUDA GPU"]
    fn test_cubecl_context() {
        if !has_cubecl_cuda_support() {
            return;
        }
        
        let device = AphelionDevice::cuda(0).unwrap();
        let ctx = CubeclContext::new(device.as_candle_device().clone()).unwrap();
        
        // Allocate output buffer
        let buffer = ctx.alloc_output(&[32, 64], candle_core::DType::F32).unwrap();
        assert_eq!(buffer.numel(), 32 * 64);
    }
}
```

### Step 11: Update Documentation

**File: `docs/getting-started.md`** (add section)

```markdown
## rust-ai-core Integration

When built with the `rust-ai-core` feature, Aphelion provides deep integration
with the rust-ai ecosystem:

### Device Management

```rust
use aphelion_core::{AphelionDevice, DeviceConfig};

// Auto-select best device (CUDA preferred)
let device = AphelionDevice::from_config(DeviceConfig::default())?;

// Explicit CUDA device
let cuda_device = AphelionDevice::cuda(0)?;

// Force CPU (with warning)
let cpu_device = AphelionDevice::cpu();
```

### Memory Tracking

```rust
use aphelion_core::{MemoryTracker, estimate_tensor_bytes, BuildContext};

// Create tracker with 8GB limit
let tracker = MemoryTracker::with_limit(8 * 1024 * 1024 * 1024);

// Estimate memory before allocation
let bytes = estimate_tensor_bytes(&[batch, seq, hidden], DType::BF16);

if tracker.would_fit(bytes) {
    tracker.allocate(bytes)?;
}

// Use in pipeline context
let ctx = BuildContext::with_memory_tracker(&backend, &trace, &tracker);
```

### CubeCL GPU Kernels

```rust
use aphelion_core::cubecl::{CubeclContext, has_cubecl_cuda_support};

if has_cubecl_cuda_support() {
    let ctx = CubeclContext::new(device)?;
    
    // Convert tensor to CubeCL buffer
    let input_buffer = ctx.tensor_to_buffer(&tensor)?;
    
    // Launch custom kernel...
    // let output_bytes = kernel.launch(&input_buffer.bytes);
    
    // Convert back to tensor
    let output = ctx.buffer_to_tensor(&output_buffer)?;
}
```
```

### Step 12: Update CI Workflow

**File: `.github/workflows/ci.yml`**

Add rust-ai-core feature testing:

```yaml
jobs:
  test:
    # ... existing config ...
    steps:
      # ... existing steps ...
      
      - name: Test with rust-ai-core
        run: cargo test --workspace --features rust-ai-core
      
      - name: Test with CUDA (if available)
        run: cargo test --workspace --features "rust-ai-core,cuda" || true
```

## Verification

After implementation, verify:

1. `cargo check --workspace --features rust-ai-core` compiles
2. `cargo test --workspace --features rust-ai-core` passes
3. `cargo test -p aphelion-python --features rust-ai-core` passes
4. Python bindings work: `maturin develop --features rust-ai-core`

```python
import aphelion

print(f"HAS_RUST_AI_CORE: {aphelion.HAS_RUST_AI_CORE}")

if aphelion.HAS_RUST_AI_CORE:
    tracker = aphelion.MemoryTracker.with_limit(1024 * 1024 * 1024)
    bytes_needed = aphelion.estimate_memory([32, 512, 768], "bf16")
    print(f"Memory needed: {bytes_needed / 1024 / 1024:.2f} MB")
    
    if tracker.would_fit(bytes_needed):
        tracker.allocate(bytes_needed)
        print(f"Allocated: {tracker.allocated_bytes() / 1024 / 1024:.2f} MB")
```

## Notes

- Keep placeholder types for when `rust-ai-core` feature is disabled
- All tensor operations go through rust-ai-core's candle interop, not direct candle usage
- MemoryTracker is optional in BuildContext (backward compatible)
- CubeCL requires both `rust-ai-core` AND `cuda` features
- Run `cargo doc --features rust-ai-core` to verify documentation
