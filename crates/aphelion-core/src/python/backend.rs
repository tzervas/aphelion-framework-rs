//! Python bindings for backend types.
//!
//! Backends abstract hardware differences (CPU, GPU, accelerators).
//! This module provides the NullBackend for testing and development.

use pyo3::prelude::*;

use crate::backend::{Backend, NullBackend};

/// Reference backend for testing and development.
///
/// NullBackend performs no actual computation. Use it for:
/// - Unit testing pipeline logic without GPU dependencies
/// - Validating graph construction and configuration
/// - Development when real hardware is unavailable
///
/// For production with GPU support, use backends from the Burn or CubeCL
/// integrations when those features are enabled.
///
/// Example:
///     >>> backend = NullBackend.cpu()
///     >>> backend.name
///     'null'
///     >>> backend.device
///     'cpu'
///
/// Attributes:
///     name (str): Backend name, always "null".
///     device (str): Device identifier (e.g., "cpu").
#[pyclass(name = "NullBackend")]
#[derive(Clone)]
pub struct PyNullBackend {
    pub(crate) inner: NullBackend,
}

#[pymethods]
impl PyNullBackend {
    /// Create a NullBackend with a specific device.
    ///
    /// Args:
    ///     device: Device identifier string (e.g., "cpu", "cuda:0").
    #[new]
    #[pyo3(text_signature = "(device)")]
    fn new(device: String) -> Self {
        Self {
            inner: NullBackend::new(device),
        }
    }

    /// Create a CPU-targeted NullBackend.
    ///
    /// Returns:
    ///     NullBackend configured for CPU.
    ///
    /// Example:
    ///     >>> backend = NullBackend.cpu()
    #[staticmethod]
    fn cpu() -> Self {
        Self {
            inner: NullBackend::cpu(),
        }
    }

    /// Backend name.
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Device identifier.
    #[getter]
    fn device(&self) -> &str {
        self.inner.device()
    }

    fn __repr__(&self) -> String {
        format!(
            "NullBackend(name='{}', device='{}')",
            self.name(),
            self.device()
        )
    }
}

/// Enum wrapper for backend types.
#[derive(Clone)]
pub enum AnyBackend {
    Null(NullBackend),
}

impl AnyBackend {
    pub fn as_backend(&self) -> &dyn Backend {
        match self {
            AnyBackend::Null(b) => b,
        }
    }
}

impl From<PyNullBackend> for AnyBackend {
    fn from(b: PyNullBackend) -> Self {
        AnyBackend::Null(b.inner)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNullBackend>()?;
    Ok(())
}
