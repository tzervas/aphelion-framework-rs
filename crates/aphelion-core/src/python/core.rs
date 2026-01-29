//! Python bindings for rust-ai-core integration.
//!
//! This module provides access to rust-ai-core's memory tracking, device
//! detection, and dtype utilities through the aphelion Python package.
//! Rather than reimplementing these utilities, we leverage and extend
//! the existing rust-ai-core Python bindings.

#![allow(clippy::useless_conversion)]

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

use rust_ai_core::memory::MemoryTracker as RustMemoryTracker;
use rust_ai_core::{get_device, DeviceConfig};

/// Memory tracker for GPU memory budget management.
///
/// Wraps rust-ai-core's MemoryTracker to provide Python access to
/// memory estimation and allocation tracking.
///
/// Example:
///     >>> tracker = MemoryTracker(limit_gb=8.0)
///     >>> tracker.allocate(1024 * 1024 * 1024)  # 1GB
///     >>> print(f"Allocated: {tracker.allocated_bytes / 1e9:.2f} GB")
///     >>> print(f"Peak: {tracker.peak_bytes / 1e9:.2f} GB")
#[pyclass(name = "MemoryTracker")]
#[derive(Clone)]
pub struct PyMemoryTracker {
    inner: Arc<Mutex<RustMemoryTracker>>,
}

#[pymethods]
impl PyMemoryTracker {
    /// Create a new memory tracker with optional limit and overhead factor.
    ///
    /// Args:
    ///     limit_gb: Memory limit in gigabytes (default: 8.0)
    ///     overhead_factor: Multiplicative overhead for memory estimates (default: 1.1)
    #[new]
    #[pyo3(signature = (limit_gb=8.0, overhead_factor=1.1))]
    fn new(limit_gb: f64, overhead_factor: f64) -> Self {
        let limit_bytes = (limit_gb * 1024.0 * 1024.0 * 1024.0) as usize;
        let tracker =
            RustMemoryTracker::with_limit(limit_bytes).with_overhead_factor(overhead_factor);
        Self {
            inner: Arc::new(Mutex::new(tracker)),
        }
    }

    /// Check if an allocation of the given size would fit within the limit.
    fn would_fit(&self, bytes: usize) -> PyResult<bool> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
        Ok(inner.would_fit(bytes))
    }

    /// Record an allocation of the given size.
    ///
    /// Raises:
    ///     RuntimeError: If the allocation would exceed the memory limit.
    fn allocate(&self, bytes: usize) -> PyResult<()> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
        inner
            .allocate(bytes)
            .map_err(|e| PyRuntimeError::new_err(format!("Allocation failed: {e}")))
    }

    /// Record a deallocation of the given size.
    fn deallocate(&self, bytes: usize) -> PyResult<()> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
        inner.deallocate(bytes);
        Ok(())
    }

    /// Reset the tracker to initial state.
    fn reset(&self) -> PyResult<()> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
        inner.reset();
        Ok(())
    }

    /// Get the current allocated bytes.
    #[getter]
    fn allocated_bytes(&self) -> PyResult<usize> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
        Ok(inner.allocated_bytes())
    }

    /// Get the peak allocated bytes.
    #[getter]
    fn peak_bytes(&self) -> PyResult<usize> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
        Ok(inner.peak_bytes())
    }

    /// Get the memory limit in bytes.
    #[getter]
    fn limit_bytes(&self) -> PyResult<usize> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
        Ok(inner.limit_bytes())
    }

    /// Estimate bytes for a tensor with overhead factor applied.
    #[pyo3(signature = (shape, dtype="f32"))]
    fn estimate_with_overhead(&self, shape: Vec<usize>, dtype: &str) -> PyResult<usize> {
        let candle_dtype = parse_dtype(dtype)?;
        let inner = self
            .inner
            .lock()
            .map_err(|e| PyRuntimeError::new_err(format!("Lock poisoned: {e}")))?;
        Ok(inner.estimate_with_overhead(&shape, candle_dtype))
    }

    fn __repr__(&self) -> String {
        let inner = self.inner.lock().ok();
        match inner {
            Some(t) => format!(
                "MemoryTracker(allocated={:.2}GB, peak={:.2}GB, limit={:.2}GB)",
                t.allocated_bytes() as f64 / 1e9,
                t.peak_bytes() as f64 / 1e9,
                t.limit_bytes() as f64 / 1e9
            ),
            None => "MemoryTracker(<locked>)".to_string(),
        }
    }
}

/// Estimate memory required for a tensor with given shape and dtype.
///
/// Args:
///     shape: Tensor shape as a list of integers
///     dtype: Data type string (f16, bf16, f32, f64, u8, u32, i16, i32, i64)
///
/// Returns:
///     Estimated bytes for the tensor
///
/// Example:
///     >>> bytes = estimate_tensor_bytes([1024, 1024], "f32")
///     >>> print(f"4MB tensor: {bytes / 1e6:.2f} MB")
#[pyfunction]
#[pyo3(signature = (shape, dtype="f32"))]
fn estimate_tensor_bytes(shape: Vec<usize>, dtype: &str) -> PyResult<usize> {
    let candle_dtype = parse_dtype(dtype)?;
    Ok(rust_ai_core::memory::estimate_tensor_bytes(
        &shape,
        candle_dtype,
    ))
}

/// Estimate memory for attention computation (O(n^2) attention scores).
///
/// Args:
///     batch_size: Batch size
///     num_heads: Number of attention heads
///     seq_len: Sequence length
///     head_dim: Dimension per head
///     dtype: Data type string (default: "bf16")
///
/// Returns:
///     Estimated bytes for attention computation
#[pyfunction]
#[pyo3(signature = (batch_size, num_heads, seq_len, head_dim, dtype="bf16"))]
fn estimate_attention_memory(
    batch_size: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    dtype: &str,
) -> PyResult<usize> {
    let candle_dtype = parse_dtype(dtype)?;
    Ok(rust_ai_core::memory::estimate_attention_memory(
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        candle_dtype,
    ))
}

/// Check if CUDA is available on this system.
///
/// Returns:
///     True if CUDA is available, False otherwise
#[pyfunction]
fn cuda_available() -> bool {
    candle_core::Device::cuda_if_available(0)
        .map(|d| matches!(d, candle_core::Device::Cuda(_)))
        .unwrap_or(false)
}

/// Get information about the available compute device.
///
/// Args:
///     force_cpu: Force CPU even if GPU is available
///     cuda_device: CUDA device ordinal to use (default: 0)
///
/// Returns:
///     Dictionary with device information:
///         - type: "cuda", "cpu", or "metal"
///         - ordinal: Device index (None for CPU)
///         - name: Human-readable device name
#[pyfunction]
#[pyo3(signature = (force_cpu=false, cuda_device=0))]
fn get_device_info(py: Python<'_>, force_cpu: bool, cuda_device: usize) -> PyResult<Py<PyAny>> {
    let config = DeviceConfig::new()
        .with_force_cpu(force_cpu)
        .with_cuda_device(cuda_device);

    let device =
        get_device(&config).map_err(|e| PyRuntimeError::new_err(format!("Device error: {e}")))?;

    let dict = pyo3::types::PyDict::new(py);

    match device {
        candle_core::Device::Cuda(_) => {
            dict.set_item("type", "cuda")?;
            dict.set_item("ordinal", cuda_device)?;
            dict.set_item("name", format!("CUDA:{cuda_device}"))?;
        }
        candle_core::Device::Cpu => {
            dict.set_item("type", "cpu")?;
            dict.set_item("ordinal", py.None())?;
            dict.set_item("name", "CPU")?;
        }
        candle_core::Device::Metal(_) => {
            dict.set_item("type", "metal")?;
            dict.set_item("ordinal", 0_usize)?;
            dict.set_item("name", "Metal:0")?;
        }
    }

    Ok(dict.into())
}

/// Get bytes per element for a data type.
///
/// Args:
///     dtype: Data type string
///
/// Returns:
///     Number of bytes per element
#[pyfunction]
fn bytes_per_dtype(dtype: &str) -> PyResult<usize> {
    let candle_dtype = parse_dtype(dtype)?;
    Ok(rust_ai_core::dtype::bytes_per_element(candle_dtype))
}

/// Check if a data type is floating point.
///
/// Args:
///     dtype: Data type string
///
/// Returns:
///     True if the dtype is floating point
#[pyfunction]
fn is_floating_point_dtype(dtype: &str) -> PyResult<bool> {
    let candle_dtype = parse_dtype(dtype)?;
    Ok(rust_ai_core::dtype::is_floating_point(candle_dtype))
}

/// Get the accumulator data type for a given dtype.
///
/// For reduced precision types (f16, bf16), returns the appropriate
/// accumulator type (f32) for numerical stability.
///
/// Args:
///     dtype: Data type string
///
/// Returns:
///     Accumulator dtype string
#[pyfunction]
fn accumulator_dtype(dtype: &str) -> PyResult<String> {
    use rust_ai_core::DTypeExt;
    let candle_dtype = parse_dtype(dtype)?;
    let acc_dtype = candle_dtype.accumulator_dtype();
    Ok(dtype_to_string(acc_dtype))
}

/// Get all supported data types.
///
/// Returns:
///     List of supported dtype strings
#[pyfunction]
fn supported_dtypes() -> Vec<&'static str> {
    vec![
        "f16", "bf16", "f32", "f64", "u8", "u32", "i16", "i32", "i64",
    ]
}

/// Get the default overhead factor for memory estimation.
///
/// Returns:
///     Default overhead factor (typically 1.1 or 10% overhead)
#[pyfunction]
fn default_overhead_factor() -> f64 {
    rust_ai_core::memory::DEFAULT_OVERHEAD_FACTOR
}

/// Get rust-ai-core version.
///
/// Returns:
///     Version string of the underlying rust-ai-core library
#[pyfunction]
fn rust_ai_core_version() -> &'static str {
    rust_ai_core::VERSION
}

fn parse_dtype(dtype: &str) -> PyResult<candle_core::DType> {
    match dtype.to_lowercase().as_str() {
        "f16" | "float16" => Ok(candle_core::DType::F16),
        "bf16" | "bfloat16" => Ok(candle_core::DType::BF16),
        "f32" | "float32" | "float" => Ok(candle_core::DType::F32),
        "f64" | "float64" | "double" => Ok(candle_core::DType::F64),
        "u8" | "uint8" => Ok(candle_core::DType::U8),
        "u32" | "uint32" => Ok(candle_core::DType::U32),
        "i16" | "int16" => Ok(candle_core::DType::I16),
        "i32" | "int32" | "int" => Ok(candle_core::DType::I32),
        "i64" | "int64" | "long" => Ok(candle_core::DType::I64),
        _ => Err(PyValueError::new_err(format!(
            "Unknown dtype: {dtype}. Supported: f16, bf16, f32, f64, u8, u32, i16, i32, i64"
        ))),
    }
}

fn dtype_to_string(dtype: candle_core::DType) -> String {
    match dtype {
        candle_core::DType::F16 => "f16".to_string(),
        candle_core::DType::BF16 => "bf16".to_string(),
        candle_core::DType::F32 => "f32".to_string(),
        candle_core::DType::F64 => "f64".to_string(),
        candle_core::DType::U8 => "u8".to_string(),
        candle_core::DType::U32 => "u32".to_string(),
        candle_core::DType::I16 => "i16".to_string(),
        candle_core::DType::I32 => "i32".to_string(),
        candle_core::DType::I64 => "i64".to_string(),
        _ => format!("{dtype:?}"),
    }
}

/// Register core module functions and classes.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Memory tracking
    m.add_class::<PyMemoryTracker>()?;
    m.add_function(wrap_pyfunction!(estimate_tensor_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_attention_memory, m)?)?;
    m.add_function(wrap_pyfunction!(default_overhead_factor, m)?)?;

    // Device detection
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_info, m)?)?;

    // Dtype utilities
    m.add_function(wrap_pyfunction!(bytes_per_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(is_floating_point_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(accumulator_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(supported_dtypes, m)?)?;

    // Version info
    m.add_function(wrap_pyfunction!(rust_ai_core_version, m)?)?;

    Ok(())
}
