//! Configuration validation example.
//!
//! This example demonstrates how to use validators to ensure
//! model configurations meet requirements:
//! - NameValidator: ensures valid model names
//! - VersionValidator: ensures semantic versioning
//! - CompositeValidator: combines multiple validators
//!
//! Validation is crucial for catching configuration errors early
//! in the build process before expensive operations.

use aphelion_core::config::ModelConfig;
use aphelion_core::validation::{
    ConfigValidator, CompositeValidator, NameValidator, VersionValidator, ValidationError,
};

/// Custom validator for checking parameter constraints.
struct ParameterValidator {
    required_params: Vec<String>,
}

impl ParameterValidator {
    fn new(required_params: Vec<String>) -> Self {
        Self { required_params }
    }
}

impl ConfigValidator for ParameterValidator {
    fn validate(&self, config: &ModelConfig) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        for param in &self.required_params {
            if !config.params.contains_key(param) {
                errors.push(ValidationError::new(
                    "parameters",
                    format!("Required parameter '{}' is missing", param),
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Custom validator for checking parameter values.
struct ParameterValueValidator {
    min_hidden_size: u32,
    max_layers: u32,
}

impl ParameterValueValidator {
    fn new(min_hidden_size: u32, max_layers: u32) -> Self {
        Self {
            min_hidden_size,
            max_layers,
        }
    }
}

impl ConfigValidator for ParameterValueValidator {
    fn validate(&self, config: &ModelConfig) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Check hidden_size parameter
        if let Some(hidden_size) = config.params.get("hidden_size").and_then(|v| v.as_u64()) {
            if (hidden_size as u32) < self.min_hidden_size {
                errors.push(ValidationError::new(
                    "hidden_size",
                    format!("Must be at least {}", self.min_hidden_size),
                ));
            }
        }

        // Check layers parameter
        if let Some(layers) = config.params.get("layers").and_then(|v| v.as_u64()) {
            if (layers as u32) > self.max_layers {
                errors.push(ValidationError::new(
                    "layers",
                    format!("Must not exceed {}", self.max_layers),
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Run the validation example.
///
/// This example demonstrates:
/// 1. Using NameValidator to validate model names
/// 2. Using VersionValidator to validate semantic versions
/// 3. Creating custom validators for domain-specific rules
/// 4. Composing multiple validators with CompositeValidator
/// 5. Handling validation errors gracefully
pub fn run_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Configuration Validation Example ===\n");

    // Example 1: Simple validators
    println!("1. Simple Validators:");

    let valid_config = ModelConfig::new("my_valid_model", "1.0.0");
    let name_result = NameValidator.validate(&valid_config);
    let version_result = VersionValidator.validate(&valid_config);

    println!("  Testing 'my_valid_model' v1.0.0");
    println!("    - Name validation: {}", if name_result.is_ok() { "PASS" } else { "FAIL" });
    println!("    - Version validation: {}", if version_result.is_ok() { "PASS" } else { "FAIL" });

    // Invalid name
    let invalid_name_config = ModelConfig::new("invalid@model!", "1.0.0");
    let name_result = NameValidator.validate(&invalid_name_config);
    println!("\n  Testing 'invalid@model!' (invalid name)");
    match name_result {
        Ok(_) => println!("    - Name validation: PASS"),
        Err(errors) => {
            println!("    - Name validation: FAIL");
            for error in errors {
                println!("      * {}", error);
            }
        }
    }

    // Invalid version
    let invalid_version_config = ModelConfig::new("my_model", "1.0");
    let version_result = VersionValidator.validate(&invalid_version_config);
    println!("\n  Testing 'my_model' v1.0 (invalid version)");
    match version_result {
        Ok(_) => println!("    - Version validation: PASS"),
        Err(errors) => {
            println!("    - Version validation: FAIL");
            for error in errors {
                println!("      * {}", error);
            }
        }
    }

    // Example 2: Composite validator
    println!("\n2. Composite Validator (Name + Version):");

    let composite = CompositeValidator::new()
        .add(Box::new(NameValidator))
        .add(Box::new(VersionValidator));

    let all_valid = ModelConfig::new("valid-model_123", "2.0.1");
    println!("\n  Testing 'valid-model_123' v2.0.1");
    match composite.validate(&all_valid) {
        Ok(_) => println!("    - Validation: PASS"),
        Err(errors) => {
            println!("    - Validation: FAIL ({} errors)", errors.len());
            for error in errors {
                println!("      * {}", error);
            }
        }
    }

    let invalid_both = ModelConfig::new("", "");
    println!("\n  Testing '' v'' (both invalid)");
    match composite.validate(&invalid_both) {
        Ok(_) => println!("    - Validation: PASS"),
        Err(errors) => {
            println!("    - Validation: FAIL ({} errors)", errors.len());
            for error in errors {
                println!("      * {}", error);
            }
        }
    }

    // Example 3: Custom validators
    println!("\n3. Custom Validators (Parameters):");

    let param_validator = ParameterValidator::new(vec![
        "hidden_size".to_string(),
        "layers".to_string(),
    ]);

    let complete_config = ModelConfig::new("my_model", "1.0.0")
        .with_param("hidden_size", serde_json::json!(256))
        .with_param("layers", serde_json::json!(4));

    println!("\n  Testing config with required parameters:");
    match param_validator.validate(&complete_config) {
        Ok(_) => println!("    - Parameter validation: PASS"),
        Err(errors) => {
            println!("    - Parameter validation: FAIL");
            for error in errors {
                println!("      * {}", error);
            }
        }
    }

    let incomplete_config = ModelConfig::new("my_model", "1.0.0")
        .with_param("hidden_size", serde_json::json!(256));

    println!("\n  Testing config missing 'layers' parameter:");
    match param_validator.validate(&incomplete_config) {
        Ok(_) => println!("    - Parameter validation: PASS"),
        Err(errors) => {
            println!("    - Parameter validation: FAIL");
            for error in errors {
                println!("      * {}", error);
            }
        }
    }

    // Example 4: Custom value validator
    println!("\n4. Custom Value Validator (Parameter Constraints):");

    let value_validator = ParameterValueValidator::new(64, 32);

    let valid_values = ModelConfig::new("my_model", "1.0.0")
        .with_param("hidden_size", serde_json::json!(128))
        .with_param("layers", serde_json::json!(16));

    println!("\n  Testing config with hidden_size=128, layers=16:");
    match value_validator.validate(&valid_values) {
        Ok(_) => println!("    - Value validation: PASS"),
        Err(errors) => {
            println!("    - Value validation: FAIL");
            for error in errors {
                println!("      * {}", error);
            }
        }
    }

    let invalid_values = ModelConfig::new("my_model", "1.0.0")
        .with_param("hidden_size", serde_json::json!(32))
        .with_param("layers", serde_json::json!(64));

    println!("\n  Testing config with hidden_size=32 (too small), layers=64 (too many):");
    match value_validator.validate(&invalid_values) {
        Ok(_) => println!("    - Value validation: PASS"),
        Err(errors) => {
            println!("    - Value validation: FAIL ({} errors)", errors.len());
            for error in errors {
                println!("      * {}", error);
            }
        }
    }

    // Example 5: Multi-level composite validator
    println!("\n5. Multi-Level Composite Validator:");

    let comprehensive = CompositeValidator::new()
        .add(Box::new(NameValidator))
        .add(Box::new(VersionValidator))
        .add(Box::new(ParameterValidator::new(vec![
            "hidden_size".to_string(),
            "layers".to_string(),
        ])))
        .add(Box::new(ParameterValueValidator::new(64, 32)));

    let good_config = ModelConfig::new("transformer-base", "1.2.3")
        .with_param("hidden_size", serde_json::json!(512))
        .with_param("layers", serde_json::json!(12));

    println!("\n  Testing comprehensive validation (valid config):");
    match comprehensive.validate(&good_config) {
        Ok(_) => println!("    - Comprehensive validation: PASS"),
        Err(errors) => {
            println!("    - Comprehensive validation: FAIL");
            for error in errors {
                println!("      * {}", error);
            }
        }
    }

    let bad_config = ModelConfig::new("bad#model", "1.0")
        .with_param("hidden_size", serde_json::json!(32));

    println!("\n  Testing comprehensive validation (invalid config):");
    match comprehensive.validate(&bad_config) {
        Ok(_) => println!("    - Comprehensive validation: PASS"),
        Err(errors) => {
            println!("    - Comprehensive validation: FAIL ({} errors)", errors.len());
            for error in errors {
                println!("      * {}", error);
            }
        }
    }

    println!("\nValidation example completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_validator() {
        let validator = ParameterValidator::new(vec!["test".to_string()]);
        let config = ModelConfig::new("test", "1.0.0");
        assert!(validator.validate(&config).is_err());
    }

    #[test]
    fn test_parameter_value_validator() {
        let validator = ParameterValueValidator::new(64, 32);
        let config = ModelConfig::new("test", "1.0.0")
            .with_param("hidden_size", serde_json::json!(128))
            .with_param("layers", serde_json::json!(16));
        assert!(validator.validate(&config).is_ok());
    }

    #[test]
    fn test_validation_example() {
        assert!(run_example().is_ok());
    }
}
