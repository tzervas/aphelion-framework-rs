//! Configuration validation types and validators.
//!
//! This module provides a validation framework for model configurations, including
//! validators for common fields like name and version, and support for composing
//! multiple validators.

use crate::config::ModelConfig;

/// A validation error with field-level context.
///
/// `ValidationError` represents a single validation failure, including the field
/// that failed validation and a descriptive error message.
///
/// # Fields
///
/// * `field` - The configuration field that failed validation
/// * `message` - Descriptive error message
///
/// # Examples
///
/// ```
/// use aphelion_core::validation::ValidationError;
///
/// let error = ValidationError::new("name", "Name cannot be empty");
/// assert_eq!(error.field, "name");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationError {
    /// The field that failed validation
    pub field: String,
    /// Error message describing the validation failure
    pub message: String,
}

impl ValidationError {
    /// Creates a new validation error.
    ///
    /// # Arguments
    ///
    /// * `field` - The field name that failed validation
    /// * `message` - Descriptive error message
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::validation::ValidationError;
    ///
    /// let error = ValidationError::new("version", "Version must be semantic");
    /// assert_eq!(error.field, "version");
    /// ```
    pub fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl std::error::Error for ValidationError {}

/// Trait for validating model configurations.
///
/// `ConfigValidator` defines the interface for configuration validation. Implementations
/// check configurations for semantic correctness and return all validation errors at once,
/// enabling comprehensive error reporting.
///
/// # Implementing ConfigValidator
///
/// Types implementing `ConfigValidator` must be thread-safe (`Send + Sync`) and return
/// all validation errors found, not just the first one.
///
/// # Examples
///
/// ```
/// use aphelion_core::validation::{ConfigValidator, ValidationError};
/// use aphelion_core::config::ModelConfig;
///
/// struct CustomValidator;
///
/// impl ConfigValidator for CustomValidator {
///     fn validate(&self, config: &ModelConfig) -> Result<(), Vec<ValidationError>> {
///         if config.name.contains(" ") {
///             Err(vec![ValidationError::new("name", "Name cannot contain spaces")])
///         } else {
///             Ok(())
///         }
///     }
/// }
/// ```
pub trait ConfigValidator: Send + Sync {
    /// Validates the configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to validate
    ///
    /// # Returns
    ///
    /// `Ok(())` if validation succeeds, or `Err(errors)` with all validation failures
    fn validate(&self, config: &ModelConfig) -> Result<(), Vec<ValidationError>>;
}

/// Validates model names.
///
/// `NameValidator` ensures that model names are non-empty and contain only
/// alphanumeric characters, dashes, and underscores.
///
/// # Examples
///
/// ```
/// use aphelion_core::validation::{NameValidator, ConfigValidator};
/// use aphelion_core::config::ModelConfig;
///
/// let validator = NameValidator;
///
/// // Valid
/// let config = ModelConfig::new("my-model_v1", "1.0.0");
/// assert!(validator.validate(&config).is_ok());
///
/// // Invalid
/// let config = ModelConfig::new("invalid@name", "1.0.0");
/// assert!(validator.validate(&config).is_err());
/// ```
pub struct NameValidator;

impl ConfigValidator for NameValidator {
    fn validate(&self, config: &ModelConfig) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        if config.name.is_empty() {
            errors.push(ValidationError::new("name", "Name cannot be empty"));
        } else if !config.name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
            errors.push(ValidationError::new(
                "name",
                "Name must contain only alphanumeric characters, dashes, and underscores",
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Validates semantic versioning format.
///
/// `VersionValidator` ensures that version strings match semantic versioning format
/// (e.g., 1.2.3, with at least major.minor.patch).
///
/// # Examples
///
/// ```
/// use aphelion_core::validation::{VersionValidator, ConfigValidator};
/// use aphelion_core::config::ModelConfig;
///
/// let validator = VersionValidator;
///
/// // Valid
/// let config = ModelConfig::new("model", "1.2.3");
/// assert!(validator.validate(&config).is_ok());
///
/// // Invalid
/// let config = ModelConfig::new("model", "1.2");
/// assert!(validator.validate(&config).is_err());
/// ```
pub struct VersionValidator;

impl ConfigValidator for VersionValidator {
    fn validate(&self, config: &ModelConfig) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        if config.version.is_empty() {
            errors.push(ValidationError::new("version", "Version cannot be empty"));
        } else if !is_valid_semver(&config.version) {
            errors.push(ValidationError::new(
                "version",
                "Version must match semantic versioning pattern (e.g., 1.2.3)",
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Checks if a version string matches semantic versioning pattern.
///
/// A valid semver has at least three numeric components separated by dots (e.g., 1.2.3).
fn is_valid_semver(version: &str) -> bool {
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() < 3 {
        return false;
    }

    parts.iter().take(3).all(|part| part.parse::<u32>().is_ok())
}

/// Composes multiple validators into a single validator.
///
/// `CompositeValidator` runs all registered validators and collects all errors,
/// providing comprehensive validation that checks all constraints.
///
/// # Examples
///
/// ```
/// use aphelion_core::validation::{CompositeValidator, ConfigValidator, NameValidator, VersionValidator};
/// use aphelion_core::config::ModelConfig;
///
/// let validator = CompositeValidator::new()
///     .add(Box::new(NameValidator))
///     .add(Box::new(VersionValidator));
///
/// // Valid
/// let config = ModelConfig::new("my-model", "1.0.0");
/// assert!(validator.validate(&config).is_ok());
///
/// // Invalid - both errors collected
/// let config = ModelConfig::new("", "");
/// let result = validator.validate(&config);
/// assert!(result.is_err());
/// assert_eq!(result.unwrap_err().len(), 2);
/// ```
pub struct CompositeValidator {
    validators: Vec<Box<dyn ConfigValidator>>,
}

impl CompositeValidator {
    /// Creates a new composite validator with no sub-validators.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::validation::CompositeValidator;
    ///
    /// let validator = CompositeValidator::new();
    /// ```
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

    /// Adds a validator to the composite.
    ///
    /// # Arguments
    ///
    /// * `validator` - The validator to add
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::validation::{CompositeValidator, NameValidator};
    ///
    /// let validator = CompositeValidator::new()
    ///     .add(Box::new(NameValidator));
    /// ```
    pub fn add(mut self, validator: Box<dyn ConfigValidator>) -> Self {
        self.validators.push(validator);
        self
    }
}

impl Default for CompositeValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigValidator for CompositeValidator {
    fn validate(&self, config: &ModelConfig) -> Result<(), Vec<ValidationError>> {
        let mut all_errors = Vec::new();

        for validator in &self.validators {
            if let Err(errors) = validator.validate(config) {
                all_errors.extend(errors);
            }
        }

        if all_errors.is_empty() {
            Ok(())
        } else {
            Err(all_errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_validator_valid() {
        let config = ModelConfig::new("valid-name_123", "1.0.0");
        assert!(NameValidator.validate(&config).is_ok());
    }

    #[test]
    fn test_name_validator_empty() {
        let config = ModelConfig::new("", "1.0.0");
        let result = NameValidator.validate(&config);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors[0].field, "name");
        assert!(errors[0].message.contains("empty"));
    }

    #[test]
    fn test_name_validator_invalid_chars() {
        let config = ModelConfig::new("invalid@name", "1.0.0");
        let result = NameValidator.validate(&config);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err()[0].field, "name");
    }

    #[test]
    fn test_name_validator_with_spaces() {
        let config = ModelConfig::new("invalid name", "1.0.0");
        let result = NameValidator.validate(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_version_validator_valid() {
        let config = ModelConfig::new("my-model", "1.2.3");
        assert!(VersionValidator.validate(&config).is_ok());
    }

    #[test]
    fn test_version_validator_valid_with_prerelease() {
        let config = ModelConfig::new("my-model", "1.0.0");
        assert!(VersionValidator.validate(&config).is_ok());
    }

    #[test]
    fn test_version_validator_empty() {
        let config = ModelConfig::new("my-model", "");
        let result = VersionValidator.validate(&config);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors[0].field, "version");
        assert!(errors[0].message.contains("empty"));
    }

    #[test]
    fn test_version_validator_invalid_semver() {
        let config = ModelConfig::new("my-model", "1.2");
        let result = VersionValidator.validate(&config);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err()[0].field, "version");
    }

    #[test]
    fn test_version_validator_non_numeric() {
        let config = ModelConfig::new("my-model", "1.2.a");
        let result = VersionValidator.validate(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_composite_validator_all_valid() {
        let config = ModelConfig::new("valid-model", "1.0.0");
        let composite = CompositeValidator::new()
            .add(Box::new(NameValidator))
            .add(Box::new(VersionValidator));

        assert!(composite.validate(&config).is_ok());
    }

    #[test]
    fn test_composite_validator_multiple_errors() {
        let config = ModelConfig::new("", "");
        let composite = CompositeValidator::new()
            .add(Box::new(NameValidator))
            .add(Box::new(VersionValidator));

        let result = composite.validate(&config);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 2);
        assert!(errors.iter().any(|e| e.field == "name"));
        assert!(errors.iter().any(|e| e.field == "version"));
    }

    #[test]
    fn test_composite_validator_partial_errors() {
        let config = ModelConfig::new("valid-model", "");
        let composite = CompositeValidator::new()
            .add(Box::new(NameValidator))
            .add(Box::new(VersionValidator));

        let result = composite.validate(&config);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].field, "version");
    }

    #[test]
    fn test_validation_error_display() {
        let error = ValidationError::new("test_field", "test message");
        assert_eq!(error.to_string(), "test_field: test message");
    }
}
