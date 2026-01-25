//! Model configuration types and traits.
//!
//! This module provides the configuration infrastructure for defining model properties,
//! parameters, and metadata. Configurations use deterministic ordering for reproducibility
//! and support semantic versioning for tracking model iterations.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Generic model configuration container with deterministic ordering.
///
/// `ModelConfig` stores model metadata including name, version, and arbitrary parameters
/// in a deterministic order (using `BTreeMap`). This ensures configurations can be
/// consistently hashed and compared across different runs.
///
/// # Fields
///
/// * `name` - The model identifier (e.g., "gpt-3-fine-tuned")
/// * `version` - Semantic version string (e.g., "1.2.3")
/// * `params` - Additional parameters as JSON values, stored in sorted key order
///
/// # Examples
///
/// ```
/// use aphelion_core::config::ModelConfig;
///
/// let config = ModelConfig::new("my-model", "1.0.0")
///     .with_param("learning_rate", serde_json::json!(0.001))
///     .with_param("batch_size", serde_json::json!(32));
///
/// assert_eq!(config.name, "my-model");
/// assert_eq!(config.version, "1.0.0");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    /// Model name or identifier
    pub name: String,
    /// Semantic version string
    pub version: String,
    /// Model parameters in deterministic order
    pub params: BTreeMap<String, serde_json::Value>,
}

impl ModelConfig {
    /// Creates a new configuration with the given name and version.
    ///
    /// The configuration starts with an empty parameter map that can be populated
    /// using the builder pattern via `with_param`.
    ///
    /// # Arguments
    ///
    /// * `name` - Model identifier (converted to String)
    /// * `version` - Semantic version string (converted to String)
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::config::ModelConfig;
    ///
    /// let config = ModelConfig::new("bert-base", "2.1.0");
    /// assert_eq!(config.name, "bert-base");
    /// assert_eq!(config.version, "2.1.0");
    /// assert!(config.params.is_empty());
    /// ```
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            params: BTreeMap::new(),
        }
    }

    /// Adds or updates a parameter in the configuration.
    ///
    /// Uses the builder pattern to chain multiple parameter additions.
    /// Parameters are stored in a `BTreeMap` to ensure deterministic ordering.
    ///
    /// # Arguments
    ///
    /// * `key` - Parameter name (converted to String)
    /// * `value` - Parameter value as a JSON value
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::config::ModelConfig;
    ///
    /// let config = ModelConfig::new("model", "1.0.0")
    ///     .with_param("dropout", serde_json::json!(0.1))
    ///     .with_param("hidden_size", serde_json::json!(768));
    ///
    /// assert_eq!(config.params.len(), 2);
    /// ```
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.params.insert(key.into(), value);
        self
    }

    /// Creates a small model preset configuration.
    ///
    /// Provides predefined parameters for a small model:
    /// - hidden_size: 256
    /// - num_layers: 2
    /// - dropout: 0.1
    /// - activation: relu
    ///
    /// # Arguments
    ///
    /// * `name` - Model identifier for the preset
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::config::ModelConfig;
    ///
    /// let config = ModelConfig::small_preset("small_bert");
    /// assert_eq!(config.params["hidden_size"], 256);
    /// ```
    pub fn small_preset(name: &str) -> Self {
        Self {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            params: vec![
                ("hidden_size".to_string(), serde_json::json!(256)),
                ("num_layers".to_string(), serde_json::json!(2)),
                ("dropout".to_string(), serde_json::json!(0.1)),
                ("activation".to_string(), serde_json::json!("relu")),
            ]
            .into_iter()
            .collect(),
        }
    }

    /// Creates a medium model preset configuration.
    ///
    /// Provides predefined parameters for a medium model:
    /// - hidden_size: 512
    /// - num_layers: 4
    /// - dropout: 0.2
    /// - activation: relu
    ///
    /// # Arguments
    ///
    /// * `name` - Model identifier for the preset
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::config::ModelConfig;
    ///
    /// let config = ModelConfig::medium_preset("medium_bert");
    /// assert_eq!(config.params["hidden_size"], 512);
    /// ```
    pub fn medium_preset(name: &str) -> Self {
        Self {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            params: vec![
                ("hidden_size".to_string(), serde_json::json!(512)),
                ("num_layers".to_string(), serde_json::json!(4)),
                ("dropout".to_string(), serde_json::json!(0.2)),
                ("activation".to_string(), serde_json::json!("relu")),
            ]
            .into_iter()
            .collect(),
        }
    }

    /// Creates a large model preset configuration.
    ///
    /// Provides predefined parameters for a large model:
    /// - hidden_size: 1024
    /// - num_layers: 8
    /// - dropout: 0.3
    /// - activation: relu
    ///
    /// # Arguments
    ///
    /// * `name` - Model identifier for the preset
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::config::ModelConfig;
    ///
    /// let config = ModelConfig::large_preset("large_bert");
    /// assert_eq!(config.params["hidden_size"], 1024);
    /// ```
    pub fn large_preset(name: &str) -> Self {
        Self {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            params: vec![
                ("hidden_size".to_string(), serde_json::json!(1024)),
                ("num_layers".to_string(), serde_json::json!(8)),
                ("dropout".to_string(), serde_json::json!(0.3)),
                ("activation".to_string(), serde_json::json!("relu")),
            ]
            .into_iter()
            .collect(),
        }
    }
}

/// Builder for constructing `ModelConfig` with a fluent API.
///
/// `ModelConfigBuilder` provides a flexible, type-safe way to construct `ModelConfig` instances
/// with specialized methods for common parameters like `hidden_size`, `num_layers`, and `dropout`.
/// The builder pattern ensures all required fields are provided before building.
///
/// # Examples
///
/// ```
/// use aphelion_core::config::ModelConfigBuilder;
///
/// let config = ModelConfigBuilder::new()
///     .name("my-model")
///     .version("1.0.0")
///     .hidden_size(512)
///     .num_layers(4)
///     .dropout(0.2)
///     .activation("gelu")
///     .build()
///     .expect("failed to build config");
///
/// assert_eq!(config.name, "my-model");
/// ```
pub struct ModelConfigBuilder {
    name: Option<String>,
    version: Option<String>,
    params: BTreeMap<String, serde_json::Value>,
}

impl ModelConfigBuilder {
    /// Creates a new empty `ModelConfigBuilder`.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::config::ModelConfigBuilder;
    ///
    /// let builder = ModelConfigBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self {
            name: None,
            version: None,
            params: BTreeMap::new(),
        }
    }

    /// Sets the model name.
    ///
    /// # Arguments
    ///
    /// * `name` - Model identifier (converted to String)
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::config::ModelConfigBuilder;
    ///
    /// let config = ModelConfigBuilder::new()
    ///     .name("bert-base")
    ///     .version("1.0.0")
    ///     .build()
    ///     .expect("failed to build");
    /// ```
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the model version.
    ///
    /// # Arguments
    ///
    /// * `version` - Semantic version string (converted to String)
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Adds a parameter with a serializable value.
    ///
    /// # Arguments
    ///
    /// * `key` - Parameter name
    /// * `value` - Parameter value (must implement `Serialize`)
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::config::ModelConfigBuilder;
    ///
    /// let config = ModelConfigBuilder::new()
    ///     .name("model")
    ///     .version("1.0.0")
    ///     .param("learning_rate", 0.001)
    ///     .param("seed", 42)
    ///     .build()
    ///     .expect("failed to build");
    /// ```
    pub fn param<T: Serialize>(mut self, key: impl Into<String>, value: T) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.params.insert(key.into(), json_value);
        }
        self
    }

    /// Sets the hidden size parameter (number of units in hidden layers).
    ///
    /// # Arguments
    ///
    /// * `size` - Hidden layer size
    pub fn hidden_size(self, size: usize) -> Self {
        self.param("hidden_size", size)
    }

    /// Sets the number of layers parameter.
    ///
    /// # Arguments
    ///
    /// * `layers` - Number of layers
    pub fn num_layers(self, layers: usize) -> Self {
        self.param("num_layers", layers)
    }

    /// Sets the dropout rate parameter.
    ///
    /// # Arguments
    ///
    /// * `rate` - Dropout probability between 0.0 and 1.0
    pub fn dropout(self, rate: f64) -> Self {
        self.param("dropout", rate)
    }

    /// Sets the activation function parameter.
    ///
    /// # Arguments
    ///
    /// * `name` - Activation function name (e.g., "relu", "gelu", "sigmoid")
    pub fn activation(self, name: &str) -> Self {
        self.param("activation", name)
    }

    /// Builds the `ModelConfig`, returning an error if required fields are missing.
    ///
    /// # Errors
    ///
    /// Returns an error string if `name` or `version` is not set.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::config::ModelConfigBuilder;
    ///
    /// // Success case
    /// let result = ModelConfigBuilder::new()
    ///     .name("model")
    ///     .version("1.0.0")
    ///     .build();
    /// assert!(result.is_ok());
    ///
    /// // Error case - missing version
    /// let result = ModelConfigBuilder::new()
    ///     .name("model")
    ///     .build();
    /// assert!(result.is_err());
    /// ```
    pub fn build(self) -> Result<ModelConfig, String> {
        let name = self
            .name
            .ok_or_else(|| "ModelConfig requires a name".to_string())?;
        let version = self
            .version
            .ok_or_else(|| "ModelConfig requires a version".to_string())?;

        Ok(ModelConfig {
            name,
            version,
            params: self.params,
        })
    }
}

impl Default for ModelConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that provide configuration specifications.
///
/// Implementors of this trait expose their configuration through the `ConfigSpec` interface,
/// enabling uniform configuration handling across different model builders and components.
/// This trait is thread-safe (`Send + Sync`) to work with Aphelion's concurrent pipeline
/// infrastructure.
///
/// # Implementing ConfigSpec
///
/// Types that need to participate in the Aphelion pipeline should implement `ConfigSpec`
/// to expose their configuration for validation, serialization, and pipeline processing.
///
/// # Examples
///
/// ```
/// use aphelion_core::config::{ConfigSpec, ModelConfig};
///
/// struct MyModel {
///     config: ModelConfig,
/// }
///
/// impl ConfigSpec for MyModel {
///     fn config(&self) -> &ModelConfig {
///         &self.config
///     }
/// }
/// ```
pub trait ConfigSpec: Send + Sync {
    /// Returns a reference to this type's configuration.
    fn config(&self) -> &ModelConfig;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let config = ModelConfigBuilder::new()
            .name("test_model")
            .version("1.0.0")
            .build()
            .expect("failed to build config");

        assert_eq!(config.name, "test_model");
        assert_eq!(config.version, "1.0.0");
        assert!(config.params.is_empty());
    }

    #[test]
    fn test_builder_with_params() {
        let config = ModelConfigBuilder::new()
            .name("test_model")
            .version("1.0.0")
            .hidden_size(256)
            .num_layers(4)
            .dropout(0.2)
            .activation("relu")
            .build()
            .expect("failed to build config");

        assert_eq!(config.name, "test_model");
        assert_eq!(config.version, "1.0.0");
        assert_eq!(config.params.len(), 4);
        assert_eq!(config.params["hidden_size"], 256);
        assert_eq!(config.params["num_layers"], 4);
        assert_eq!(config.params["dropout"], 0.2);
        assert_eq!(config.params["activation"], "relu");
    }

    #[test]
    fn test_builder_missing_name() {
        let result = ModelConfigBuilder::new()
            .version("1.0.0")
            .build();

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "ModelConfig requires a name");
    }

    #[test]
    fn test_builder_missing_version() {
        let result = ModelConfigBuilder::new()
            .name("test_model")
            .build();

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "ModelConfig requires a version");
    }

    #[test]
    fn test_builder_generic_param() {
        let config = ModelConfigBuilder::new()
            .name("test_model")
            .version("1.0.0")
            .param("custom_param", 42)
            .param("another_param", "value")
            .build()
            .expect("failed to build config");

        assert_eq!(config.params["custom_param"], 42);
        assert_eq!(config.params["another_param"], "value");
    }

    #[test]
    fn test_builder_default() {
        let builder = ModelConfigBuilder::default();
        let result = builder.name("test").version("1.0").build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_small_preset() {
        let config = ModelConfig::small_preset("small_model");

        assert_eq!(config.name, "small_model");
        assert_eq!(config.version, "1.0.0");
        assert_eq!(config.params["hidden_size"], 256);
        assert_eq!(config.params["num_layers"], 2);
        assert_eq!(config.params["dropout"], 0.1);
        assert_eq!(config.params["activation"], "relu");
    }

    #[test]
    fn test_medium_preset() {
        let config = ModelConfig::medium_preset("medium_model");

        assert_eq!(config.name, "medium_model");
        assert_eq!(config.version, "1.0.0");
        assert_eq!(config.params["hidden_size"], 512);
        assert_eq!(config.params["num_layers"], 4);
        assert_eq!(config.params["dropout"], 0.2);
        assert_eq!(config.params["activation"], "relu");
    }

    #[test]
    fn test_large_preset() {
        let config = ModelConfig::large_preset("large_model");

        assert_eq!(config.name, "large_model");
        assert_eq!(config.version, "1.0.0");
        assert_eq!(config.params["hidden_size"], 1024);
        assert_eq!(config.params["num_layers"], 8);
        assert_eq!(config.params["dropout"], 0.3);
        assert_eq!(config.params["activation"], "relu");
    }

    #[test]
    fn test_builder_chaining() {
        let config = ModelConfigBuilder::new()
            .name("chained_model")
            .version("2.0.0")
            .hidden_size(512)
            .hidden_size(768) // Override with second call
            .build()
            .expect("failed to build config");

        assert_eq!(config.params["hidden_size"], 768);
    }

    #[test]
    fn test_builder_with_complex_values() {
        let config = ModelConfigBuilder::new()
            .name("complex_model")
            .version("1.0.0")
            .param("layers", vec![128, 256, 512])
            .build()
            .expect("failed to build config");

        let layers = &config.params["layers"];
        assert_eq!(layers[0], 128);
        assert_eq!(layers[1], 256);
        assert_eq!(layers[2], 512);
    }

    #[test]
    fn test_model_config_new() {
        let config = ModelConfig::new("test", "1.0");
        assert_eq!(config.name, "test");
        assert_eq!(config.version, "1.0");
        assert!(config.params.is_empty());
    }

    #[test]
    fn test_model_config_with_param() {
        let config = ModelConfig::new("test", "1.0")
            .with_param("key1", serde_json::json!(42))
            .with_param("key2", serde_json::json!("value"));

        assert_eq!(config.params["key1"], 42);
        assert_eq!(config.params["key2"], "value");
    }

    #[test]
    fn test_preset_sizes() {
        let small = ModelConfig::small_preset("small");
        let medium = ModelConfig::medium_preset("medium");
        let large = ModelConfig::large_preset("large");

        // Verify scaling progression
        let small_hidden: usize = small.params["hidden_size"].as_u64().unwrap() as usize;
        let medium_hidden: usize = medium.params["hidden_size"].as_u64().unwrap() as usize;
        let large_hidden: usize = large.params["hidden_size"].as_u64().unwrap() as usize;

        assert!(small_hidden < medium_hidden);
        assert!(medium_hidden < large_hidden);

        let small_layers: usize = small.params["num_layers"].as_u64().unwrap() as usize;
        let medium_layers: usize = medium.params["num_layers"].as_u64().unwrap() as usize;
        let large_layers: usize = large.params["num_layers"].as_u64().unwrap() as usize;

        assert!(small_layers < medium_layers);
        assert!(medium_layers < large_layers);
    }
}
