//! Python bindings for validation.
//!
//! Composable validators for checking configuration correctness before
//! pipeline execution. Validators can be combined to enforce multiple rules.

use pyo3::prelude::*;

use crate::config::PyModelConfig;
use aphelion_core::validation::{
    ConfigValidator, NameValidator, ValidationError, VersionValidator,
};

/// Validation error describing what failed and why.
///
/// Each error identifies the problematic field and provides a human-readable
/// message explaining the issue.
///
/// Example:
///     >>> errors = validator.validate(config)
///     >>> for err in errors:
///     ...     print(f"{err.field}: {err.message}")
///     version: Invalid semantic version format
///
/// Attributes:
///     field (str): Name of the invalid field.
///     message (str): Description of what's wrong.
#[pyclass(name = "ValidationError")]
#[derive(Clone)]
pub struct PyValidationError {
    pub(crate) inner: ValidationError,
}

#[pymethods]
impl PyValidationError {
    /// Create a validation error.
    ///
    /// Args:
    ///     field: Name of the invalid field.
    ///     message: Description of the error.
    #[new]
    #[pyo3(text_signature = "(field, message)")]
    fn new(field: String, message: String) -> Self {
        Self {
            inner: ValidationError::new(field, message),
        }
    }

    /// Name of the invalid field.
    #[getter]
    fn field(&self) -> &str {
        &self.inner.field
    }

    /// Error description.
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

/// Validator for model component names.
///
/// Ensures names are non-empty and contain only valid characters.
/// Valid names: alphanumeric, underscores, hyphens.
///
/// Example:
///     >>> validator = NameValidator()
///     >>> errors = validator.validate(config)
///     >>> if not errors:
///     ...     print("Name is valid")
#[pyclass(name = "NameValidator")]
pub struct PyNameValidator;

#[pymethods]
impl PyNameValidator {
    /// Create a name validator.
    #[new]
    fn new() -> Self {
        Self
    }

    /// Validate a configuration's name field.
    ///
    /// Args:
    ///     config: ModelConfig to validate.
    ///
    /// Returns:
    ///     List of ValidationErrors (empty if valid).
    #[pyo3(text_signature = "(config)")]
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

/// Validator for semantic version strings.
///
/// Ensures versions follow semver format: major.minor.patch
/// Accepts: "1.0.0", "2.1.3", "0.1.0-alpha"
/// Rejects: "1.0", "v1.0.0", "latest"
///
/// Example:
///     >>> validator = VersionValidator()
///     >>> errors = validator.validate(config)
#[pyclass(name = "VersionValidator")]
pub struct PyVersionValidator;

#[pymethods]
impl PyVersionValidator {
    /// Create a version validator.
    #[new]
    fn new() -> Self {
        Self
    }

    /// Validate a configuration's version field.
    ///
    /// Args:
    ///     config: ModelConfig to validate.
    ///
    /// Returns:
    ///     List of ValidationErrors (empty if valid).
    #[pyo3(text_signature = "(config)")]
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

/// Composite validator combining multiple validation rules.
///
/// Build validators incrementally using the builder pattern.
/// All validators run, collecting all errors rather than failing fast.
///
/// Why composite validators:
/// - Single validation call for multiple rules
/// - Consistent error collection
/// - Extensible with custom validators
///
/// Example:
///     >>> validator = CompositeValidator()
///     >>> validator = validator.with_name_validator()
///     >>> validator = validator.with_version_validator()
///     >>> errors = validator.validate(config)
///     >>> if errors:
///     ...     for err in errors:
///     ...         print(err)
#[pyclass(name = "CompositeValidator")]
pub struct PyCompositeValidator {
    validators: Vec<String>,
}

#[pymethods]
impl PyCompositeValidator {
    /// Create an empty composite validator.
    ///
    /// Use with_*_validator() methods to add validation rules.
    #[new]
    fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

    /// Add name validation.
    ///
    /// Returns:
    ///     New validator with name validation added.
    fn with_name_validator(&self) -> Self {
        let mut validators = self.validators.clone();
        validators.push("name".to_string());
        Self { validators }
    }

    /// Add version validation.
    ///
    /// Returns:
    ///     New validator with version validation added.
    fn with_version_validator(&self) -> Self {
        let mut validators = self.validators.clone();
        validators.push("version".to_string());
        Self { validators }
    }

    /// Run all validators on a configuration.
    ///
    /// All validators execute regardless of earlier failures, collecting
    /// all errors for comprehensive feedback.
    ///
    /// Args:
    ///     config: ModelConfig to validate.
    ///
    /// Returns:
    ///     List of all ValidationErrors (empty if valid).
    #[pyo3(text_signature = "(config)")]
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
