//! WASM bindings for validation types.

use super::config::JsModelConfig;
use crate::validation::{ConfigValidator, NameValidator, ValidationError, VersionValidator};
use wasm_bindgen::prelude::*;

/// A validation error with field and message.
#[wasm_bindgen]
pub struct JsValidationError {
    inner: ValidationError,
}

#[wasm_bindgen]
impl JsValidationError {
    /// Create a new validation error.
    #[wasm_bindgen(constructor)]
    pub fn new(field: &str, message: &str) -> Self {
        Self {
            inner: ValidationError::new(field.to_string(), message.to_string()),
        }
    }

    /// Get the field that failed validation.
    #[wasm_bindgen(getter)]
    pub fn field(&self) -> String {
        self.inner.field.clone()
    }

    /// Get the error message.
    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.inner.message.clone()
    }

    /// Convert to string representation.
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_js(&self) -> String {
        format!("{}: {}", self.inner.field, self.inner.message)
    }
}

impl JsValidationError {
    pub(crate) fn from_inner(inner: ValidationError) -> Self {
        Self { inner }
    }
}

/// Validator for model names.
///
/// Validates that names contain only alphanumeric characters,
/// underscores, and hyphens, and are not empty.
#[wasm_bindgen]
pub struct JsNameValidator;

#[wasm_bindgen]
impl JsNameValidator {
    /// Create a new name validator.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self
    }

    /// Validate a model configuration's name.
    #[wasm_bindgen]
    pub fn validate(&self, config: &JsModelConfig) -> Vec<JsValidationError> {
        let validator = NameValidator;
        match validator.validate(config.inner()) {
            Ok(()) => Vec::new(),
            Err(errors) => errors
                .into_iter()
                .map(JsValidationError::from_inner)
                .collect(),
        }
    }
}

impl Default for JsNameValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validator for semantic versions.
///
/// Validates that versions follow the semver format (major.minor.patch).
#[wasm_bindgen]
pub struct JsVersionValidator;

#[wasm_bindgen]
impl JsVersionValidator {
    /// Create a new version validator.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self
    }

    /// Validate a model configuration's version.
    #[wasm_bindgen]
    pub fn validate(&self, config: &JsModelConfig) -> Vec<JsValidationError> {
        let validator = VersionValidator;
        match validator.validate(config.inner()) {
            Ok(()) => Vec::new(),
            Err(errors) => errors
                .into_iter()
                .map(JsValidationError::from_inner)
                .collect(),
        }
    }
}

impl Default for JsVersionValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Composite validator that combines multiple validators.
#[wasm_bindgen]
pub struct JsCompositeValidator {
    use_name: bool,
    use_version: bool,
}

#[wasm_bindgen]
impl JsCompositeValidator {
    /// Create an empty composite validator.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            use_name: false,
            use_version: false,
        }
    }

    /// Add name validation.
    #[wasm_bindgen(js_name = withNameValidator)]
    pub fn with_name_validator(&self) -> JsCompositeValidator {
        Self {
            use_name: true,
            use_version: self.use_version,
        }
    }

    /// Add version validation.
    #[wasm_bindgen(js_name = withVersionValidator)]
    pub fn with_version_validator(&self) -> JsCompositeValidator {
        Self {
            use_name: self.use_name,
            use_version: true,
        }
    }

    /// Validate a model configuration.
    #[wasm_bindgen]
    pub fn validate(&self, config: &JsModelConfig) -> Vec<JsValidationError> {
        let mut errors = Vec::new();

        if self.use_name {
            let validator = NameValidator;
            if let Err(errs) = validator.validate(config.inner()) {
                errors.extend(errs.into_iter().map(JsValidationError::from_inner));
            }
        }

        if self.use_version {
            let validator = VersionValidator;
            if let Err(errs) = validator.validate(config.inner()) {
                errors.extend(errs.into_iter().map(JsValidationError::from_inner));
            }
        }

        errors
    }
}

impl Default for JsCompositeValidator {
    fn default() -> Self {
        Self::new()
    }
}
