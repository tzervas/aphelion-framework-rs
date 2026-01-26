# Getting Started with Aphelion Framework

Welcome to Aphelion. This guide covers installation, core concepts, and common patterns for using the Aphelion Framework as a frontend for AI model development in Rust.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts Walkthrough](#core-concepts-walkthrough)
   - [Creating a ModelConfig](#creating-a-modelconfig)
   - [Building a Graph](#building-a-graph)
   - [Using the Pipeline](#using-the-pipeline)
   - [Working with Backends](#working-with-backends)
   - [Capturing Traces](#capturing-traces)
5. [Feature Flags Guide](#feature-flags-guide)
6. [Common Patterns](#common-patterns)
7. [Troubleshooting](#troubleshooting)
8. [Next Steps](#next-steps)

---

## Prerequisites

Before getting started, ensure you have the following:

### Rust Toolchain

- **Rust version**: 1.70.0 or later (edition 2021)
- **Cargo**: Comes with Rust installation

To check your Rust version:

```bash
rustc --version
```

To install or update Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
```

### Required Knowledge

- Basic understanding of Rust programming
- Familiarity with Cargo and `Cargo.toml` dependencies
- Understanding of DAG (Directed Acyclic Graph) concepts is helpful but not required

### Optional

- **Tokio runtime**: For async pipeline execution (enabled via feature flag)
- **serde_json**: Required for parameter values in configurations

---

## Installation

### Basic Installation

Add `aphelion-core` to your `Cargo.toml`:

```toml
[dependencies]
aphelion-core = "1.2"
serde_json = "1.0"  # Required for parameter values
```

### With Optional Features

Enable additional functionality with feature flags:

```toml
[dependencies]
aphelion-core = { version = "1.0", features = ["tokio"] }
serde_json = "1.0"
```

### With rust-ai-core Integration

```toml
[dependencies]
aphelion-core = { version = "1.2", features = ["rust-ai-core", "tokio"] }
serde_json = "1.0"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

### Full Installation (All Features)

```toml
[dependencies]
aphelion-core = { version = "1.2", features = ["rust-ai-core", "burn", "cubecl", "tokio"] }
serde_json = "1.0"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

### From Git (Latest Development)

```toml
[dependencies]
aphelion-core = { git = "https://github.com/tzervas/aphelion-framework-rs", branch = "main" }
serde_json = "1.0"
```

---

## Quick Start

Here's a minimal working example to get you started immediately:

```rust
use aphelion_core::prelude::*;

fn main() -> AphelionResult<()> {
    // 1. Create a model configuration
    let config = ModelConfig::new("my-first-model", "1.0.0")
        .with_param("hidden_size", serde_json::json!(256))
        .with_param("num_layers", serde_json::json!(4));

    // 2. Build a computation graph
    let mut graph = BuildGraph::default();
    let input = graph.add_node("input", config.clone());
    let hidden = graph.add_node("hidden", config.clone());
    let output = graph.add_node("output", config);
    
    graph.add_edge(input, hidden);
    graph.add_edge(hidden, output);

    // 3. Set up backend and tracing
    let backend = NullBackend::cpu();
    let trace = InMemoryTraceSink::new();
    let ctx = BuildContext::new(&backend, &trace);

    // 4. Execute the pipeline
    let pipeline = BuildPipeline::new();
    let result = pipeline.execute(&ctx, graph)?;

    // 5. Get the deterministic hash
    println!("Graph hash: {}", result.stable_hash());
    println!("Nodes: {}, Edges: {}", result.node_count(), result.edge_count());

    Ok(())
}
```

Run with:

```bash
cargo run
```

---

## Core Concepts Walkthrough

### Creating a ModelConfig

`ModelConfig` is the foundation for storing model parameters in a type-safe, serializable format.

#### Basic Configuration

```rust
use aphelion_core::config::ModelConfig;

// Create a simple config with name and version
let config = ModelConfig::new("transformer", "1.0.0");
```

#### Adding Parameters

Use the builder pattern to add parameters:

```rust
use aphelion_core::config::ModelConfig;

let config = ModelConfig::new("transformer", "2.0.0")
    .with_param("d_model", serde_json::json!(512))
    .with_param("n_heads", serde_json::json!(8))
    .with_param("n_layers", serde_json::json!(6))
    .with_param("dropout", serde_json::json!(0.1))
    .with_param("vocab_size", serde_json::json!(50000));

// Access the configuration
println!("Model: {} v{}", config.name, config.version);
println!("Parameters: {:?}", config.params);
```

#### Type-Safe Parameter Retrieval

```rust
use aphelion_core::config::ModelConfig;

let config = ModelConfig::new("model", "1.0.0")
    .with_param("hidden_size", serde_json::json!(256))
    .with_param("learning_rate", serde_json::json!(0.001));

// Get required parameter (returns AphelionResult)
let hidden_size: u32 = config.param("hidden_size")?;

// Get optional parameter with default
let dropout: f64 = config.param_or("dropout", 0.0)?;

// Get parameter with type's default value
let batch_size: u32 = config.param_or_default("batch_size")?;
```

### Building a Graph

`BuildGraph` represents your model architecture as a Directed Acyclic Graph (DAG).

#### Creating Nodes and Edges

```rust
use aphelion_core::prelude::*;

let mut graph = BuildGraph::default();

// Add nodes - each returns a NodeId for reference
let encoder = graph.add_node("encoder", ModelConfig::new("encoder", "1.0.0"));
let attention = graph.add_node("attention", ModelConfig::new("attention", "1.0.0"));
let decoder = graph.add_node("decoder", ModelConfig::new("decoder", "1.0.0"));

// Add edges to define data flow (from -> to)
graph.add_edge(encoder, attention);
graph.add_edge(attention, decoder);
```

#### Deterministic Hashing

The graph produces a deterministic hash regardless of construction order:

```rust
use aphelion_core::prelude::*;

let mut graph = BuildGraph::default();
let a = graph.add_node("a", ModelConfig::new("layer", "1.0.0"));
let b = graph.add_node("b", ModelConfig::new("layer", "1.0.0"));
graph.add_edge(a, b);

// SHA-256 hash for reproducibility and caching
let hash = graph.stable_hash();
println!("Graph hash: {}", hash);

// Same graph structure = same hash (every time)
```

#### Querying the Graph

```rust
// Get node and edge counts
println!("Nodes: {}", graph.node_count());
println!("Edges: {}", graph.edge_count());

// Access nodes directly
for node in &graph.nodes {
    println!("Node {}: {} ({})", 
        node.id.value(), 
        node.name, 
        node.config.name
    );
}
```

### Using the Pipeline

`BuildPipeline` orchestrates the execution of your model build process.

#### Basic Pipeline Execution

```rust
use aphelion_core::prelude::*;

fn run_pipeline() -> AphelionResult<()> {
    // Set up context
    let backend = NullBackend::cpu();
    let trace = InMemoryTraceSink::new();
    let ctx = BuildContext::new(&backend, &trace);

    // Create graph
    let mut graph = BuildGraph::default();
    graph.add_node("layer", ModelConfig::new("model", "1.0.0"));

    // Execute pipeline
    let pipeline = BuildPipeline::new();
    let result = pipeline.execute(&ctx, graph)?;

    Ok(())
}
```

#### Pipeline with Custom Stages

```rust
use aphelion_core::prelude::*;
use aphelion_core::pipeline::PipelineStage;
use aphelion_core::diagnostics::TraceEvent;
use std::time::SystemTime;

// Define a custom stage
struct LoggingStage;

impl PipelineStage for LoggingStage {
    fn name(&self) -> &str {
        "logging"
    }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph) -> AphelionResult<()> {
        ctx.trace.record(TraceEvent {
            id: "custom.logging".to_string(),
            message: format!("Processing {} nodes", graph.node_count()),
            timestamp: SystemTime::now(),
        });
        Ok(())
    }
}

// Use in pipeline
let pipeline = BuildPipeline::new()
    .with_stage(Box::new(LoggingStage));
```

#### Pipeline with Hooks and Progress

```rust
use aphelion_core::prelude::*;

let pipeline = BuildPipeline::new()
    .with_pre_hook(|ctx| {
        println!("Starting build on backend: {}", ctx.backend.name());
        Ok(())
    })
    .with_post_hook(|_ctx, graph| {
        println!("Build complete! Hash: {}", graph.stable_hash());
        Ok(())
    })
    .with_progress(|stage_name, current, total| {
        println!("[{}/{}] Executing: {}", current, total, stage_name);
    });
```

#### Preset Pipelines

```rust
use aphelion_core::prelude::*;

// Standard pipeline (validation + hashing)
let standard = BuildPipeline::standard();

// Training-optimized pipeline
let training = BuildPipeline::for_training();

// Inference-optimized pipeline (minimal, fast)
let inference = BuildPipeline::for_inference();
```

### Working with Backends

Backends abstract hardware details and provide a consistent interface.

#### Using the NullBackend

For testing and development:

```rust
use aphelion_core::backend::NullBackend;

// CPU backend
let cpu_backend = NullBackend::cpu();

// Custom device name
let custom_backend = NullBackend::new("my-device");

println!("Backend: {}", cpu_backend.name());  // "null"
println!("Device: {}", cpu_backend.device()); // "cpu"
```

#### Implementing a Custom Backend

```rust
use aphelion_core::backend::{Backend, DeviceCapabilities};
use aphelion_core::error::AphelionResult;

struct MyGpuBackend {
    device_id: u32,
}

impl Backend for MyGpuBackend {
    fn name(&self) -> &str {
        "my_gpu"
    }

    fn device(&self) -> &str {
        "cuda:0"
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            supports_f16: true,
            supports_bf16: true,
            supports_tf32: true,
            max_memory_bytes: Some(8 * 1024 * 1024 * 1024), // 8GB
            compute_units: Some(2560),
        }
    }

    fn is_available(&self) -> bool {
        true
    }

    fn initialize(&mut self) -> AphelionResult<()> {
        println!("Initializing GPU {}...", self.device_id);
        Ok(())
    }

    fn shutdown(&mut self) -> AphelionResult<()> {
        println!("Shutting down GPU...");
        Ok(())
    }
}
```

#### Using Backend Capabilities

```rust
use aphelion_core::backend::Backend;

fn check_backend_features(backend: &dyn Backend) {
    let caps = backend.capabilities();
    
    if caps.supports_bf16 {
        println!("BF16 supported - can use brain floating point!");
    }
    
    if let Some(memory) = caps.max_memory_bytes {
        println!("Available memory: {} GB", memory / (1024 * 1024 * 1024));
    }
}
```

### Capturing Traces

Tracing provides structured logging for debugging and observability.

#### Basic Tracing

```rust
use aphelion_core::diagnostics::{InMemoryTraceSink, TraceEvent, TraceSink};
use std::time::SystemTime;

let trace = InMemoryTraceSink::new();

// Record events
trace.record(TraceEvent {
    id: "model.init".to_string(),
    message: "Initializing model".to_string(),
    timestamp: SystemTime::now(),
});

trace.record(TraceEvent {
    id: "model.build".to_string(),
    message: "Building graph complete".to_string(),
    timestamp: SystemTime::now(),
});

// Retrieve all events
for event in trace.events() {
    println!("[{}] {}", event.id, event.message);
}
```

#### Using TraceSinkExt Helpers

```rust
use aphelion_core::diagnostics::{InMemoryTraceSink, TraceSinkExt};

let trace = InMemoryTraceSink::new();

// Convenient helper methods
trace.info("model.start", "Starting model build");
trace.warn("config.deprecated", "Parameter 'old_name' is deprecated");
trace.error("validation.failed", "Invalid configuration detected");
```

#### Exporting Traces to JSON

```rust
use aphelion_core::diagnostics::InMemoryTraceSink;

let trace = InMemoryTraceSink::new();
// ... record events ...

// Export to JSON for external analysis
let json = trace.to_json();
println!("{}", json);

// Or serialize events manually
let events = trace.events();
let json = serde_json::to_string_pretty(&events)?;
```

---

## Feature Flags Guide

Aphelion uses feature flags to keep the core library lightweight while allowing optional integrations.

| Feature | Description | Use Case |
|---------|-------------|----------|
| `default` | Core functionality only | Basic pipeline building |
| `rust-ai-core` | Memory tracking, device detection, dtype utilities via rust-ai-core | Production deployments |
| `cuda` | CUDA GPU support (requires `rust-ai-core`) | NVIDIA GPU acceleration |
| `burn` | Burn deep learning framework integration | Neural network training/inference |
| `cubecl` | CubeCL GPU compute backend | GPU compute |
| `tokio` | Async pipeline execution | Concurrent operations |
| `tritter-accel` | Tritter accelerator support | Specialized hardware |

### Enabling Features

```toml
# Single feature
[dependencies]
aphelion-core = { version = "1.0", features = ["tokio"] }

# Multiple features
[dependencies]
aphelion-core = { version = "1.0", features = ["burn", "tokio"] }

# All features (for development/testing)
[dependencies]
aphelion-core = { version = "1.0", features = ["burn", "cubecl", "tokio"] }
```

### Async Execution (tokio feature)

```rust
// Requires: features = ["tokio"]
use aphelion_core::prelude::*;

#[tokio::main]
async fn main() -> AphelionResult<()> {
    let backend = NullBackend::cpu();
    let trace = InMemoryTraceSink::new();
    let ctx = BuildContext::new(&backend, &trace);

    let pipeline = BuildPipeline::new()
        .with_async_stage(Box::new(ValidationStage));

    let mut graph = BuildGraph::default();
    graph.add_node("model", ModelConfig::new("async-model", "1.0.0"));

    let result = pipeline.execute_async(&ctx, graph).await?;
    Ok(())
}
```

---

## Common Patterns

### Pattern 1: Model Builder Trait

Implement `ModelBuilder` for your custom models:

```rust
use aphelion_core::backend::{Backend, ModelBuilder};
use aphelion_core::config::{ConfigSpec, ModelConfig};
use aphelion_core::diagnostics::TraceSink;
use aphelion_core::graph::BuildGraph;

struct TransformerModel {
    config: ModelConfig,
}

impl TransformerModel {
    fn new(layers: u32, hidden_size: u32) -> Self {
        Self {
            config: ModelConfig::new("transformer", "1.0.0")
                .with_param("layers", serde_json::json!(layers))
                .with_param("hidden_size", serde_json::json!(hidden_size)),
        }
    }
}

impl ConfigSpec for TransformerModel {
    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

impl ModelBuilder for TransformerModel {
    type Output = BuildGraph;

    fn build(&self, _backend: &dyn Backend, _trace: &dyn TraceSink) -> BuildGraph {
        let mut graph = BuildGraph::default();
        
        let layers: u32 = self.config.params["layers"].as_u64().unwrap() as u32;
        let mut prev_id = None;
        
        for i in 0..layers {
            let id = graph.add_node(
                format!("layer_{}", i),
                self.config.clone()
            );
            if let Some(prev) = prev_id {
                graph.add_edge(prev, id);
            }
            prev_id = Some(id);
        }
        
        graph
    }
}
```

### Pattern 2: Validation Before Build

Always validate configurations early:

```rust
use aphelion_core::validation::{CompositeValidator, NameValidator, VersionValidator};
use aphelion_core::config::ModelConfig;

fn build_validated_model(config: &ModelConfig) -> AphelionResult<BuildGraph> {
    // Validate first
    let validator = CompositeValidator::new()
        .with_validator(Box::new(NameValidator))
        .with_validator(Box::new(VersionValidator));

    match validator.validate(config) {
        Ok(_) => {},
        Err(errors) => {
            for error in &errors {
                eprintln!("Validation error: {}", error);
            }
            return Err(AphelionError::validation("Configuration validation failed"));
        }
    }

    // Build graph after validation
    let mut graph = BuildGraph::default();
    graph.add_node("validated_model", config.clone());
    Ok(graph)
}
```

### Pattern 3: Reusable Pipeline Configuration

Create preset pipeline configurations:

```rust
use aphelion_core::prelude::*;
use aphelion_core::pipeline::PipelineStage;

fn create_production_pipeline() -> BuildPipeline {
    BuildPipeline::new()
        .with_stage(Box::new(ValidationStage))
        .with_stage(Box::new(OptimizationStage))
        .with_stage(Box::new(HashingStage))
        .with_pre_hook(|ctx| {
            ctx.trace.info("pipeline", "Production build starting");
            Ok(())
        })
        .with_post_hook(|ctx, graph| {
            ctx.trace.info("pipeline", &format!("Build hash: {}", graph.stable_hash()));
            Ok(())
        })
}
```

### Pattern 4: Conditional Stage Execution

Skip stages based on conditions:

```rust
use aphelion_core::prelude::*;
use aphelion_core::pipeline::PipelineStage;

struct ConditionalStage {
    enabled: bool,
}

impl PipelineStage for ConditionalStage {
    fn name(&self) -> &str {
        "conditional"
    }

    fn should_skip(&self, _ctx: &BuildContext, _graph: &BuildGraph) -> bool {
        !self.enabled
    }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph) -> AphelionResult<()> {
        ctx.trace.info("conditional", "Executing conditional stage");
        Ok(())
    }
}
```

### Pattern 5: Graph Hash Caching

Use deterministic hashes for caching:

```rust
use aphelion_core::prelude::*;
use std::collections::HashMap;

struct BuildCache {
    cache: HashMap<String, BuildGraph>,
}

impl BuildCache {
    fn get_or_build<F>(&mut self, graph: &BuildGraph, build_fn: F) -> &BuildGraph
    where
        F: FnOnce() -> BuildGraph,
    {
        let hash = graph.stable_hash();
        
        self.cache.entry(hash).or_insert_with(|| {
            println!("Cache miss - building...");
            build_fn()
        })
    }
}
```

---

## Troubleshooting

### Common Issues

#### 1. "serde_json not found"

**Problem**: Missing `serde_json` dependency.

**Solution**: Add to `Cargo.toml`:

```toml
[dependencies]
serde_json = "1.0"
```

#### 2. "Cannot find type ModelConfig in prelude"

**Problem**: Import issue.

**Solution**: Use the correct import:

```rust
// Either use the prelude
use aphelion_core::prelude::*;

// Or import directly
use aphelion_core::config::ModelConfig;
```

#### 3. "Async functions require tokio feature"

**Problem**: Using async without the feature flag.

**Solution**: Enable the tokio feature:

```toml
[dependencies]
aphelion-core = { version = "1.0", features = ["tokio"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

#### 4. "Graph hash changes unexpectedly"

**Problem**: Non-deterministic graph construction.

**Solution**: Ensure you're not using HashMap (use BTreeMap instead) and that node addition order is consistent:

```rust
// ✗ Bad - HashMap has non-deterministic iteration order
use std::collections::HashMap;

// ✓ Good - BTreeMap maintains insertion order
use std::collections::BTreeMap;
```

#### 5. "Validation errors not showing all issues"

**Problem**: Validator returns on first error.

**Solution**: Use `CompositeValidator` to collect all errors:

```rust
use aphelion_core::validation::CompositeValidator;

let validator = CompositeValidator::new()
    .with_validator(Box::new(NameValidator))
    .with_validator(Box::new(VersionValidator));

// Returns ALL validation errors at once
match validator.validate(&config) {
    Ok(_) => println!("Valid!"),
    Err(errors) => {
        for error in errors {
            eprintln!("Error: {}", error);
        }
    }
}
```

### Debugging Tips

1. **Enable tracing**: Always use `InMemoryTraceSink` during development to capture events.

2. **Check graph structure**: Print node/edge counts to verify graph construction.

3. **Verify hashes**: Use `stable_hash()` to ensure determinism.

4. **Use validation stages**: Add `ValidationStage` to catch config errors early.

5. **Run with RUST_BACKTRACE**: For detailed error traces:

```bash
RUST_BACKTRACE=1 cargo run
```

---

## Next Steps

Now that you understand the basics, explore these resources:

### Documentation

- [API Guide](API-GUIDE.md) - Comprehensive patterns and advanced usage
- [Architecture](ARCHITECTURE.md) - System design and module responsibilities
- [Specification](../SPEC.md) - Success criteria and compliance checklist

### Examples

Run the included examples:

```bash
# Basic usage walkthrough
cargo run --package aphelion-examples --example basic_usage

# Custom backend implementation
cargo run --package aphelion-examples --example custom_backend

# Pipeline stages and hooks
cargo run --package aphelion-examples --example pipeline_stages

# Configuration validation
cargo run --package aphelion-examples --example validation
```

### Testing

Run the test suite to see more usage patterns:

```bash
# All tests
cargo test --workspace

# With all features
cargo test --workspace --all-features

# Specific test
cargo test --package aphelion-core -- config
```

### Community

- **Repository**: https://github.com/tzervas/aphelion-framework-rs
- **Issues**: Report bugs or request features
- **Pull Requests**: Contributions welcome!

---

## Quick Reference

```rust
// Import everything you need
use aphelion_core::prelude::*;

// Core types
ModelConfig::new("name", "version")    // Configuration
BuildGraph::default()                   // Graph container
BuildPipeline::new()                    // Pipeline builder
NullBackend::cpu()                      // Test backend
InMemoryTraceSink::new()               // Trace collector
BuildContext::new(&backend, &trace)     // Execution context

// Graph operations
graph.add_node("name", config)          // Returns NodeId
graph.add_edge(from_id, to_id)          // Connect nodes
graph.stable_hash()                     // Get deterministic hash

// Pipeline execution
pipeline.execute(&ctx, graph)?          // Run pipeline
pipeline.execute_async(&ctx, graph).await?  // Async (tokio feature)

// Result type
AphelionResult<T>                       // = Result<T, AphelionError>
```

Happy building with Aphelion.
