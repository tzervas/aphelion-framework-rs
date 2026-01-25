//! Integration stubs for rust-ai-core.

#[cfg(feature = "rust-ai-core")]
pub mod adapter {
    use crate::config::ModelConfig;
    use crate::graph::BuildGraph;

    pub fn to_rust_ai_core(_graph: &BuildGraph, _config: &ModelConfig) {
        // TODO: map aphelion-core types to rust-ai-core equivalents.
    }
}
