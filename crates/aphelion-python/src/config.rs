//! Python bindings for ModelConfig.

use pyo3::prelude::*;
use std::collections::BTreeMap;

use aphelion_core::config::ModelConfig;

/// Python wrapper for ModelConfig.
#[pyclass(name = "ModelConfig")]
#[derive(Clone)]
pub struct PyModelConfig {
    pub(crate) inner: ModelConfig,
}

#[pymethods]
impl PyModelConfig {
    #[new]
    fn new(name: String, version: String) -> Self {
        Self {
            inner: ModelConfig::new(name, version),
        }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn version(&self) -> &str {
        &self.inner.version
    }

    #[getter]
    fn params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let result = pythonize::pythonize(py, &self.inner.params).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to convert params: {e}"))
        })?;
        Ok(result.unbind())
    }

    fn with_param(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let json_value: serde_json::Value = pythonize::depythonize(value).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to convert value: {e}"))
        })?;
        self.inner = self.inner.clone().with_param(key, json_value);
        Ok(self.clone())
    }

    fn param(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match self.inner.params.get(key) {
            Some(v) => {
                let result = pythonize::pythonize(py, v).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Failed to convert value: {e}"))
                })?;
                Ok(result.unbind())
            }
            None => Ok(py.None()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelConfig(name='{}', version='{}', params={})",
            self.inner.name,
            self.inner.version,
            self.inner.params.len()
        )
    }
}

/// Builder for ModelConfig.
#[pyclass(name = "ModelConfigBuilder")]
pub struct PyModelConfigBuilder {
    name: Option<String>,
    version: Option<String>,
    params: BTreeMap<String, serde_json::Value>,
}

#[pymethods]
impl PyModelConfigBuilder {
    #[new]
    fn new() -> Self {
        Self {
            name: None,
            version: None,
            params: BTreeMap::new(),
        }
    }

    fn name(&mut self, name: String) -> Self {
        Self {
            name: Some(name),
            version: self.version.clone(),
            params: self.params.clone(),
        }
    }

    fn version(&mut self, version: String) -> Self {
        Self {
            name: self.name.clone(),
            version: Some(version),
            params: self.params.clone(),
        }
    }

    fn param(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let json_value: serde_json::Value = pythonize::depythonize(value).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to convert value: {e}"))
        })?;
        let mut params = self.params.clone();
        params.insert(key.to_string(), json_value);
        Ok(Self {
            name: self.name.clone(),
            version: self.version.clone(),
            params,
        })
    }

    fn build(&self) -> PyResult<PyModelConfig> {
        let name = self.name.clone()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("name is required"))?;
        let version = self.version.clone()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("version is required"))?;

        let mut config = ModelConfig::new(name, version);
        for (k, v) in &self.params {
            config = config.with_param(k, v.clone());
        }
        Ok(PyModelConfig { inner: config })
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModelConfig>()?;
    m.add_class::<PyModelConfigBuilder>()?;
    Ok(())
}
