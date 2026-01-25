//! Python bindings for backend types.

use pyo3::prelude::*;

use aphelion_core::backend::{Backend, NullBackend};

/// Reference CPU-only backend for testing.
#[pyclass(name = "NullBackend")]
#[derive(Clone)]
pub struct PyNullBackend {
    pub(crate) inner: NullBackend,
}

#[pymethods]
impl PyNullBackend {
    #[new]
    fn new(device: String) -> Self {
        Self {
            inner: NullBackend::new(device),
        }
    }

    #[staticmethod]
    fn cpu() -> Self {
        Self {
            inner: NullBackend::cpu(),
        }
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

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
