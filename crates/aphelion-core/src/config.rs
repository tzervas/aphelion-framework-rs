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
}

/// Simple versioned configuration trait for model builders.
pub trait ConfigSpec: Send + Sync {
    fn config(&self) -> &ModelConfig;
}
