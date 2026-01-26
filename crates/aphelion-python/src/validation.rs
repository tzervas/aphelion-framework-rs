//! Python bindings for validation.

use pyo3::prelude::*;

use crate::config::PyModelConfig;
use aphelion_core::validation::{
    ConfigValidator, NameValidator, ValidationError, VersionValidator,
};

/// Validation error with field and message.
#[pyclass(name = "ValidationError")]
#[derive(Clone)]
pub struct PyValidationError {
    pub(crate) inner: ValidationError,
}

#[pymethods]
impl PyValidationError {
    #[new]
    fn new(field: String, message: String) -> Self {
        Self {
            inner: ValidationError::new(field, message),
        }
    }

    #[getter]
    fn field(&self) -> &str {
        &self.inner.field
    }

    #[getter]
    fn message(&self) -> &str {
        &self.inner.message
    }

    fn __repr__(&self) -> String {
        format!(
            "ValidationError(field='{}', message='{}')",
            self.inner.field, self.inner.message
        )
    }

    fn __str__(&self) -> String {
        format!("{}: {}", self.inner.field, self.inner.message)
    }
}

/// Validator for model names.
#[pyclass(name = "NameValidator")]
pub struct PyNameValidator;

#[pymethods]
impl PyNameValidator {
    #[new]
    fn new() -> Self {
        Self
    }

    fn validate(&self, config: &PyModelConfig) -> Vec<PyValidationError> {
        match NameValidator.validate(&config.inner) {
            Ok(()) => Vec::new(),
            Err(errors) => errors
                .into_iter()
                .map(|e| PyValidationError { inner: e })
                .collect(),
        }
    }

    fn __repr__(&self) -> &str {
        "NameValidator"
    }
}

/// Validator for semantic versions.
#[pyclass(name = "VersionValidator")]
pub struct PyVersionValidator;

#[pymethods]
impl PyVersionValidator {
    #[new]
    fn new() -> Self {
        Self
    }

    fn validate(&self, config: &PyModelConfig) -> Vec<PyValidationError> {
        match VersionValidator.validate(&config.inner) {
            Ok(()) => Vec::new(),
            Err(errors) => errors
                .into_iter()
                .map(|e| PyValidationError { inner: e })
                .collect(),
        }
    }

    fn __repr__(&self) -> &str {
        "VersionValidator"
    }
}

/// Composite validator that runs multiple validators.
#[pyclass(name = "CompositeValidator")]
pub struct PyCompositeValidator {
    validators: Vec<String>,
}

#[pymethods]
impl PyCompositeValidator {
    #[new]
    fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

    fn with_name_validator(&self) -> Self {
        let mut validators = self.validators.clone();
        validators.push("name".to_string());
        Self { validators }
    }

    fn with_version_validator(&self) -> Self {
        let mut validators = self.validators.clone();
        validators.push("version".to_string());
        Self { validators }
    }

    fn validate(&self, config: &PyModelConfig) -> Vec<PyValidationError> {
        let mut errors = Vec::new();

        for validator_name in &self.validators {
            match validator_name.as_str() {
                "name" => {
                    if let Err(errs) = NameValidator.validate(&config.inner) {
                        errors.extend(errs.into_iter().map(|e| PyValidationError { inner: e }));
                    }
                }
                "version" => {
                    if let Err(errs) = VersionValidator.validate(&config.inner) {
                        errors.extend(errs.into_iter().map(|e| PyValidationError { inner: e }));
                    }
                }
                _ => {}
            }
        }

        errors
    }

    fn __repr__(&self) -> String {
        format!("CompositeValidator(validators={:?})", self.validators)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyValidationError>()?;
    m.add_class::<PyNameValidator>()?;
    m.add_class::<PyVersionValidator>()?;
    m.add_class::<PyCompositeValidator>()?;
    Ok(())
}
