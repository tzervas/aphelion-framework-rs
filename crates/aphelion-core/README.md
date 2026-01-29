# aphelion-core

Core library for the [Aphelion Framework](https://github.com/tzervas/aphelion-framework-rs) - a unified frontend for AI model development in Rust.

## Overview

`aphelion-core` provides the foundational APIs for building AI pipelines:

- **BuildGraph** - Directed acyclic graph for model architecture with deterministic SHA-256 hashing
- **BuildPipeline** - Composable pipeline stages for training, inference, and deployment
- **ModelConfig** - Type-safe configuration with parameter validation and versioning
- **Backend** - Hardware abstraction trait for CPU, GPU, and accelerators
- **Diagnostics** - Structured tracing and event logging

## Installation

```toml
[dependencies]
aphelion-core = "1.2"
serde_json = "1.0"  # Required for parameter values
```

### Optional Features

```toml
[dependencies]
aphelion-core = { version = "1.2", features = ["rust-ai-core", "tritter-accel", "tokio"] }
```

| Feature | Description |
|---------|-------------|
| `rust-ai-core` | Memory tracking, device detection, dtype utilities via [rust-ai-core](https://crates.io/crates/rust-ai-core) |
| `tritter-accel` | BitNet b1.58 ternary ops, VSA gradient compression via [tritter-accel](https://crates.io/crates/tritter-accel) |
| `cuda` | CUDA GPU support (requires `rust-ai-core`) |
| `burn` | Burn deep learning framework backend |
| `cubecl` | CubeCL GPU compute backend |
| `tokio` | Async pipeline execution |
| `python` | Python bindings via PyO3 (builds `aphelion-framework` wheel) |

## Quick Start

```rust
use aphelion_core::prelude::*;
use aphelion_core::config::ModelConfig;
use aphelion_core::backend::NullBackend;
use aphelion_core::diagnostics::InMemoryTraceSink;
use aphelion_core::graph::BuildGraph;
use aphelion_core::pipeline::{BuildContext, BuildPipeline};

// Create model configuration
let config = ModelConfig::new("transformer", "1.0.0")
    .with_param("d_model", serde_json::json!(512))
    .with_param("n_heads", serde_json::json!(8));

// Build graph
let mut graph = BuildGraph::default();
let node = graph.add_node("encoder", config);

// Execute pipeline
let backend = NullBackend::cpu();
let trace = InMemoryTraceSink::new();
let ctx = BuildContext::new(&backend, &trace);

let pipeline = BuildPipeline::standard();
let result = pipeline.execute(&ctx, graph).unwrap();

println!("Hash: {}", result.stable_hash());
```

## Core Modules

### `config` - Model Configuration

```rust
use aphelion_core::config::ModelConfig;

let config = ModelConfig::new("llama", "2.0.0")
    .with_param("hidden_size", serde_json::json!(4096))
    .with_param("num_layers", serde_json::json!(32));

// Type-safe retrieval
let hidden: u32 = config.param("hidden_size")?;
let layers: u32 = config.param_or("num_layers", 12)?;
```

### `graph` - Build Graph

```rust
use aphelion_core::graph::BuildGraph;

let mut graph = BuildGraph::default();
let input = graph.add_node("input", config.clone());
let hidden = graph.add_node("hidden", config.clone());
graph.add_edge(input, hidden);

// Deterministic hash
let hash = graph.stable_hash();
```

### `pipeline` - Pipeline Execution

```rust
use aphelion_core::pipeline::{BuildPipeline, ValidationStage, HashingStage};

let pipeline = BuildPipeline::new()
    .with_stage(Box::new(ValidationStage))
    .with_stage(Box::new(HashingStage))
    .with_pre_hook(|ctx| {
        println!("Starting on {}", ctx.backend.name());
        Ok(())
    });

let result = pipeline.execute(&ctx, graph)?;
```

### `backend` - Hardware Abstraction

```rust
use aphelion_core::backend::{Backend, NullBackend, DeviceCapabilities};

// Use null backend for testing
let backend = NullBackend::cpu();

// Implement custom backend
impl Backend for MyGpuBackend {
    fn name(&self) -> &str { "my_gpu" }
    fn device(&self) -> &str { "cuda:0" }
    fn capabilities(&self) -> DeviceCapabilities { /* ... */ }
    fn is_available(&self) -> bool { true }
    fn initialize(&mut self) -> AphelionResult<()> { Ok(()) }
    fn shutdown(&mut self) -> AphelionResult<()> { Ok(()) }
}
```

### `diagnostics` - Tracing

```rust
use aphelion_core::diagnostics::{InMemoryTraceSink, TraceSinkExt};

let trace = InMemoryTraceSink::new();
trace.info("model.init", "Initializing model");
trace.warn("config", "Deprecated parameter");

let json = trace.to_json();
```

## Ecosystem Integration

aphelion-core is part of the rust-ai ecosystem:

- [rust-ai-core](https://crates.io/crates/rust-ai-core) - Memory tracking, device detection
- [tritter-accel](https://crates.io/crates/tritter-accel) - Ternary acceleration
- [Candle](https://github.com/huggingface/candle) - Tensor operations

See the [framework README](https://github.com/tzervas/aphelion-framework-rs) for full ecosystem documentation.

## License

MIT License - see [LICENSE](https://github.com/tzervas/aphelion-framework-rs/blob/main/LICENSE)
