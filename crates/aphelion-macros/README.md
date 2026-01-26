# aphelion-macros

Procedural macros for the [Aphelion Framework](https://github.com/tzervas/aphelion-framework-rs) - a unified frontend for AI model development in Rust.

## Overview

`aphelion-macros` provides the `#[aphelion_model]` attribute macro for ergonomic model definitions with automatic trait implementations.

## Installation

```toml
[dependencies]
aphelion-macros = "1.2"
```

> **Note**: This crate is automatically included when you depend on `aphelion-core`. You typically don't need to add it directly.

## Usage

### `#[aphelion_model]` Attribute

Apply to structs that contain a `ModelConfig` field to automatically derive common traits:

```rust
use aphelion_core::aphelion_model;
use aphelion_core::config::ModelConfig;

#[aphelion_model]
pub struct TransformerModel {
    pub config: ModelConfig,
}

impl TransformerModel {
    pub fn new(d_model: u32, n_heads: u32) -> Self {
        Self {
            config: ModelConfig::new("transformer", "1.0.0")
                .with_param("d_model", serde_json::json!(d_model))
                .with_param("n_heads", serde_json::json!(n_heads)),
        }
    }
}

// Automatically derives: Debug, Clone
let model = TransformerModel::new(512, 8);
println!("{:?}", model);
```

### Generated Code

The `#[aphelion_model]` macro generates:

```rust
// Input
#[aphelion_model]
pub struct MyModel {
    pub config: ModelConfig,
}

// Generated output
#[derive(Debug, Clone)]
pub struct MyModel {
    pub config: ModelConfig,
}
```

### When to Use

Use `#[aphelion_model]` when:

- Defining model structs that wrap `ModelConfig`
- You want consistent trait implementations across models
- Building reusable model templates

For simple cases, you can also manually derive traits:

```rust
#[derive(Debug, Clone)]
pub struct SimpleModel {
    pub config: ModelConfig,
}
```

## Example: Custom Model with Methods

```rust
use aphelion_core::aphelion_model;
use aphelion_core::config::ModelConfig;
use aphelion_core::graph::BuildGraph;
use aphelion_core::backend::Backend;
use aphelion_core::diagnostics::TraceSink;

#[aphelion_model]
pub struct LlamaModel {
    pub config: ModelConfig,
}

impl LlamaModel {
    pub fn llama_7b() -> Self {
        Self {
            config: ModelConfig::new("llama", "7B")
                .with_param("hidden_size", serde_json::json!(4096))
                .with_param("num_heads", serde_json::json!(32))
                .with_param("num_layers", serde_json::json!(32))
                .with_param("vocab_size", serde_json::json!(32000)),
        }
    }

    pub fn build_graph(&self, backend: &dyn Backend, trace: &dyn TraceSink) -> BuildGraph {
        let mut graph = BuildGraph::default();
        graph.add_node("llama", self.config.clone());
        graph
    }
}
```

## Ecosystem

This crate is part of the Aphelion Framework:

- [aphelion-core](https://crates.io/crates/aphelion-core) - Core library
- [aphelion-framework-rs](https://github.com/tzervas/aphelion-framework-rs) - Full framework

## License

MIT License - see [LICENSE](https://github.com/tzervas/aphelion-framework-rs/blob/main/LICENSE)
