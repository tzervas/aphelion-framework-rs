use crate::config::ModelConfig;

/// Represents a validation error with a field name and error message.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
}

impl ValidationError {
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
pub trait ConfigValidator: Send + Sync {
    fn validate(&self, config: &ModelConfig) -> Result<(), Vec<ValidationError>>;
}

/// Validates that the model name is non-empty and contains only alphanumeric characters, dashes, and underscores.
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

/// Validates that the version matches semantic versioning pattern (e.g., 1.2.3).
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

/// Validates that a version string matches semantic versioning pattern.
fn is_valid_semver(version: &str) -> bool {
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() < 3 {
        return false;
    }

    parts.iter().take(3).all(|part| part.parse::<u32>().is_ok())
}

/// Combines multiple validators into a single validator.
pub struct CompositeValidator {
    validators: Vec<Box<dyn ConfigValidator>>,
}

impl CompositeValidator {
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

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
