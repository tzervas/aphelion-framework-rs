//! WASM bindings for ModelConfig.

use crate::config::ModelConfig;
use wasm_bindgen::prelude::*;

/// Type-safe configuration for AI model components.
///
/// ModelConfig holds the name, semantic version, and parameters for a model
/// component. Parameters are stored in sorted order for deterministic hashing.
#[wasm_bindgen]
pub struct JsModelConfig {
    inner: ModelConfig,
}

#[wasm_bindgen]
impl JsModelConfig {
    /// Create a new ModelConfig.
    #[wasm_bindgen(constructor)]
    pub fn new(name: &str, version: &str) -> Self {
        Self {
            inner: ModelConfig::new(name.to_string(), version.to_string()),
        }
    }

    /// Get the component name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Get the semantic version.
    #[wasm_bindgen(getter)]
    pub fn version(&self) -> String {
        self.inner.version.clone()
    }

    /// Get all parameters as a JSON object.
    #[wasm_bindgen(getter)]
    pub fn params(&self) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(&self.inner.params)
            .map_err(|e| JsError::new(&format!("Failed to serialize params: {}", e)))
    }

    /// Add or update a parameter. Returns a new config for chaining.
    #[wasm_bindgen(js_name = withParam)]
    pub fn with_param(&self, key: &str, value: JsValue) -> Result<JsModelConfig, JsError> {
        let json_value: serde_json::Value = serde_wasm_bindgen::from_value(value)
            .map_err(|e| JsError::new(&format!("Failed to convert value: {}", e)))?;
        Ok(Self {
            inner: self.inner.clone().with_param(key, json_value),
        })
    }

    /// Get a parameter value by key.
    #[wasm_bindgen]
    pub fn param(&self, key: &str) -> Result<JsValue, JsError> {
        match self.inner.params.get(key) {
            Some(v) => serde_wasm_bindgen::to_value(v)
                .map_err(|e| JsError::new(&format!("Failed to serialize value: {}", e))),
            None => Ok(JsValue::UNDEFINED),
        }
    }

    /// Create a small preset configuration.
    #[wasm_bindgen(js_name = smallPreset)]
    pub fn small_preset(name: &str) -> Self {
        Self {
            inner: ModelConfig::small_preset(name),
        }
    }

    /// Create a medium preset configuration.
    #[wasm_bindgen(js_name = mediumPreset)]
    pub fn medium_preset(name: &str) -> Self {
        Self {
            inner: ModelConfig::medium_preset(name),
        }
    }

    /// Create a large preset configuration.
    #[wasm_bindgen(js_name = largePreset)]
    pub fn large_preset(name: &str) -> Self {
        Self {
            inner: ModelConfig::large_preset(name),
        }
    }

    /// Serialize the config to JSON.
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsError> {
        serde_json::to_string(&self.inner)
            .map_err(|e| JsError::new(&format!("Failed to serialize: {}", e)))
    }

    /// Deserialize a config from JSON.
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(json: &str) -> Result<JsModelConfig, JsError> {
        let inner: ModelConfig = serde_json::from_str(json)
            .map_err(|e| JsError::new(&format!("Failed to deserialize: {}", e)))?;
        Ok(Self { inner })
    }
}

impl JsModelConfig {
    /// Get the inner ModelConfig (for internal use).
    pub(crate) fn inner(&self) -> &ModelConfig {
        &self.inner
    }

    /// Create from inner ModelConfig.
    pub(crate) fn from_inner(inner: ModelConfig) -> Self {
        Self { inner }
    }
}
