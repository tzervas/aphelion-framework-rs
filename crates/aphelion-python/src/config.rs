//! Python bindings for ModelConfig.
//!
//! Provides type-safe configuration management for AI model components.
//! Configurations use semantic versioning and deterministic parameter ordering.

use pyo3::prelude::*;
use std::collections::BTreeMap;

use aphelion_core::config::ModelConfig;

/// Type-safe configuration for AI model components.
///
/// ModelConfig holds the name, semantic version, and parameters for a model
/// component. Parameters are stored in a BTreeMap for deterministic iteration
/// order, ensuring identical configurations produce identical hashes.
///
/// Example:
///     >>> config = ModelConfig("transformer", "1.0.0")
///     >>> config = config.with_param("d_model", 512)
///     >>> config = config.with_param("n_heads", 8)
///     >>> config.param("d_model")
///     512
///
/// Attributes:
///     name (str): Component name (e.g., "encoder", "transformer").
///     version (str): Semantic version string (e.g., "1.0.0", "2.1.3").
///     params (dict): Configuration parameters as key-value pairs.
#[pyclass(name = "ModelConfig")]
#[derive(Clone)]
pub struct PyModelConfig {
    pub(crate) inner: ModelConfig,
}

#[pymethods]
impl PyModelConfig {
    /// Create a new ModelConfig.
    ///
    /// Args:
    ///     name: Component name identifying this model part.
    ///     version: Semantic version string (major.minor.patch).
    ///
    /// Returns:
    ///     A new ModelConfig with empty parameters.
    ///
    /// Example:
    ///     >>> config = ModelConfig("encoder", "1.0.0")
    #[new]
    #[pyo3(text_signature = "(name, version)")]
    fn new(name: String, version: String) -> Self {
        Self {
            inner: ModelConfig::new(name, version),
        }
    }

    /// Component name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Semantic version string.
    #[getter]
    fn version(&self) -> &str {
        &self.inner.version
    }

    /// Configuration parameters as a dictionary.
    #[getter]
    fn params(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let result = pythonize::pythonize(py, &self.inner.params).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to convert params: {e}"))
        })?;
        Ok(result.unbind())
    }

    /// Add or update a parameter.
    ///
    /// Returns self for method chaining. Parameters are stored in sorted order
    /// for deterministic hashing.
    ///
    /// Args:
    ///     key: Parameter name.
    ///     value: Parameter value (any JSON-serializable Python object).
    ///
    /// Returns:
    ///     Self with the parameter added.
    ///
    /// Example:
    ///     >>> config = ModelConfig("enc", "1.0.0")
    ///     >>> config = config.with_param("hidden_size", 768)
    #[pyo3(text_signature = "(key, value)")]
    fn with_param(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let json_value: serde_json::Value = pythonize::depythonize(value).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to convert value: {e}"))
        })?;
        self.inner = self.inner.clone().with_param(key, json_value);
        Ok(self.clone())
    }

    /// Get a parameter value by key.
    ///
    /// Args:
    ///     key: Parameter name to retrieve.
    ///
    /// Returns:
    ///     The parameter value, or None if not found.
    ///
    /// Example:
    ///     >>> config.param("hidden_size")
    ///     768
    #[pyo3(text_signature = "(key)")]
    fn param(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
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

/// Builder for constructing ModelConfig incrementally.
///
/// Use this when you need to construct a config in multiple steps or when
/// configuration values come from different sources.
///
/// Example:
///     >>> builder = ModelConfigBuilder()
///     >>> builder = builder.name("transformer")
///     >>> builder = builder.version("2.0.0")
///     >>> builder = builder.param("layers", 12)
///     >>> config = builder.build()
#[pyclass(name = "ModelConfigBuilder")]
pub struct PyModelConfigBuilder {
    name: Option<String>,
    version: Option<String>,
    params: BTreeMap<String, serde_json::Value>,
}

#[pymethods]
impl PyModelConfigBuilder {
    /// Create an empty builder.
    #[new]
    fn new() -> Self {
        Self {
            name: None,
            version: None,
            params: BTreeMap::new(),
        }
    }

    /// Set the component name.
    ///
    /// Args:
    ///     name: Component name.
    ///
    /// Returns:
    ///     Builder with name set.
    #[pyo3(text_signature = "(name)")]
    fn name(&mut self, name: String) -> Self {
        Self {
            name: Some(name),
            version: self.version.clone(),
            params: self.params.clone(),
        }
    }

    /// Set the semantic version.
    ///
    /// Args:
    ///     version: Semantic version string.
    ///
    /// Returns:
    ///     Builder with version set.
    #[pyo3(text_signature = "(version)")]
    fn version(&mut self, version: String) -> Self {
        Self {
            name: self.name.clone(),
            version: Some(version),
            params: self.params.clone(),
        }
    }

    /// Add a configuration parameter.
    ///
    /// Args:
    ///     key: Parameter name.
    ///     value: Parameter value.
    ///
    /// Returns:
    ///     Builder with parameter added.
    #[pyo3(text_signature = "(key, value)")]
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

    /// Build the final ModelConfig.
    ///
    /// Returns:
    ///     ModelConfig with all builder values.
    ///
    /// Raises:
    ///     ValueError: If name or version is not set.
    fn build(&self) -> PyResult<PyModelConfig> {
        let name = self
            .name
            .clone()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("name is required"))?;
        let version = self
            .version
            .clone()
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
