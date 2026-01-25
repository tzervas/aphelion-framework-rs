use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Generic model configuration container with deterministic ordering.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    pub name: String,
    pub version: String,
    pub params: BTreeMap<String, serde_json::Value>,
}

impl ModelConfig {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            params: BTreeMap::new(),
        }
    }

    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.params.insert(key.into(), value);
        self
    }

    /// Create a small model preset configuration.
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

    /// Create a medium model preset configuration.
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

    /// Create a large model preset configuration.
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

/// Builder for constructing ModelConfig with a fluent API.
pub struct ModelConfigBuilder {
    name: Option<String>,
    version: Option<String>,
    params: BTreeMap<String, serde_json::Value>,
}

impl ModelConfigBuilder {
    /// Create a new ModelConfigBuilder.
    pub fn new() -> Self {
        Self {
            name: None,
            version: None,
            params: BTreeMap::new(),
        }
    }

    /// Set the model name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the model version.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Add a parameter with a serializable value.
    pub fn param<T: Serialize>(mut self, key: impl Into<String>, value: T) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.params.insert(key.into(), json_value);
        }
        self
    }

    /// Set the hidden size parameter.
    pub fn hidden_size(self, size: usize) -> Self {
        self.param("hidden_size", size)
    }

    /// Set the number of layers parameter.
    pub fn num_layers(self, layers: usize) -> Self {
        self.param("num_layers", layers)
    }

    /// Set the dropout rate parameter.
    pub fn dropout(self, rate: f64) -> Self {
        self.param("dropout", rate)
    }

    /// Set the activation function parameter.
    pub fn activation(self, name: &str) -> Self {
        self.param("activation", name)
    }

    /// Build the ModelConfig, returning an error if required fields are missing.
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

/// Simple versioned configuration trait for model builders.
pub trait ConfigSpec: Send + Sync {
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
