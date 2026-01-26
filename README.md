# Aphelion Framework

A unified frontend for AI model development in Rust.

## Overview

Aphelion is a framework frontend that provides an easier entrypoint for AI engineering tasks. It unifies access to Rust AI libraries through a consistent API, handling the complexity of model configuration, pipeline orchestration, and hardware abstraction so you can focus on building models.

### Why Aphelion

Building AI systems in Rust means integrating multiple libraries: tensor operations, memory management, device handling, training loops. Aphelion provides:

- **Unified API**: One consistent interface to underlying AI libraries
- **Reusable Components**: Configurations and pipelines can be templated and shared
- **Deterministic Builds**: SHA-256 hashing ensures identical configs produce identical models
- **Configuration Management**: Type-safe parameters with validation and versioning
- **Pipeline Orchestration**: Composable stages for training, inference, and deployment
- **Backend Abstraction**: Write once, run on CPU, GPU, or accelerators
- **Diagnostics**: Structured tracing for debugging and observability

### Ecosystem Integration

Aphelion integrates with the Rust AI ecosystem, providing a frontend to:

| Library | Purpose | Feature Flag |
|---------|---------|--------------|
| [rust-ai-core](https://crates.io/crates/rust-ai-core) | Memory tracking, device detection, dtype utilities | `rust-ai-core` |
| [Candle](https://github.com/huggingface/candle) | Tensor operations and model inference | via `rust-ai-core` |
| [Burn](https://github.com/tracel-ai/burn) | Deep learning framework | `burn` |
| [CubeCL](https://github.com/tracel-ai/cubecl) | GPU compute | `cubecl` |

Enable features based on your needs. The framework handles library initialization, device selection, and resource management.

## Installation

### Rust

```toml
[dependencies]
aphelion-core = "1.2"
serde_json = "1.0"  # Required for parameter values
```

Optional features:

```toml
[dependencies]
aphelion-core = { version = "1.2", features = ["rust-ai-core", "tokio"] }
```

| Feature | Description |
|---------|-------------|
| `rust-ai-core` | Memory tracking, device detection, dtype utilities via [rust-ai-core](https://crates.io/crates/rust-ai-core) |
| `cuda` | CUDA GPU support (requires `rust-ai-core`) |
| `burn` | Burn deep learning framework backend (placeholder) |
| `cubecl` | CubeCL GPU compute backend (placeholder) |
| `tokio` | Async pipeline execution support |

### Python

```bash
pip install aphelion-framework
```

For memory tracking and device detection:

```bash
pip install aphelion-framework rust-ai-core-bindings
```

Python usage:

```python
import aphelion

# Build configuration
config = aphelion.ModelConfig("transformer", "1.0.0")
config = config.with_param("d_model", 512)
config = config.with_param("n_heads", 8)

# Build graph
graph = aphelion.BuildGraph()
node = graph.add_node("encoder", config)

# Execute pipeline
ctx = aphelion.BuildContext.with_null_backend()
pipeline = aphelion.BuildPipeline.standard()
result = pipeline.execute(ctx, graph)

print(f"Hash: {result.stable_hash()}")
```

## Core Concepts

### Build Graph

A directed acyclic graph representing model architecture. Each node contains configuration and metadata.

```rust
use aphelion_core::prelude::*;

let mut graph = BuildGraph::default();

// Add nodes with configuration
let input = graph.add_node("input", ModelConfig::new("encoder", "1.0.0"));
let hidden = graph.add_node("hidden", ModelConfig::new("transformer", "1.0.0"));
let output = graph.add_node("output", ModelConfig::new("decoder", "1.0.0"));

// Define data flow
graph.add_edge(input, hidden);
graph.add_edge(hidden, output);

// Deterministic hash for reproducibility
let hash = graph.stable_hash();
```

The graph hash is computed using SHA-256 over canonicalized node data. Identical graphs produce identical hashes regardless of construction order.

### Configuration

Type-safe configuration with parameter validation:

```rust
use aphelion_core::config::ModelConfig;

let config = ModelConfig::new("transformer", "2.0.0")
    .with_param("d_model", serde_json::json!(512))
    .with_param("n_heads", serde_json::json!(8))
    .with_param("n_layers", serde_json::json!(6))
    .with_param("dropout", serde_json::json!(0.1));

// Type-safe retrieval
let d_model: u32 = config.param("d_model")?;
let dropout: f64 = config.param_or("dropout", 0.0)?;
let batch_size: u32 = config.param_or_default("batch_size")?; // Uses Default::default()
```

### Pipeline Execution

Pipelines execute stages in sequence with optional hooks:

```rust
use aphelion_core::prelude::*;
use aphelion_core::pipeline::{ValidationStage, HashingStage};

// Create execution context
let backend = NullBackend::cpu();
let trace = InMemoryTraceSink::new();
let ctx = BuildContext::new(&backend, &trace);

// Build pipeline
let pipeline = BuildPipeline::new()
    .with_stage(Box::new(ValidationStage))
    .with_stage(Box::new(HashingStage))
    .with_pre_hook(|ctx| {
        println!("Starting build on {}", ctx.backend.name());
        Ok(())
    })
    .with_post_hook(|_ctx, graph| {
        println!("Completed. Hash: {}", graph.stable_hash());
        Ok(())
    })
    .with_progress(|stage, current, total| {
        println!("[{}/{}] {}", current, total, stage);
    });

// Execute
let result = pipeline.execute(&ctx, graph)?;
```

Preset pipelines for common workflows:

```rust
let pipeline = BuildPipeline::standard();      // Validation + hashing
let pipeline = BuildPipeline::for_training();  // + training hooks
let pipeline = BuildPipeline::for_inference(); // Minimal, hashing only
```

### Custom Pipeline Stages

Implement the `PipelineStage` trait:

```rust
use aphelion_core::pipeline::{PipelineStage, BuildContext};
use aphelion_core::graph::BuildGraph;
use aphelion_core::error::AphelionResult;

struct OptimizationStage;

impl PipelineStage for OptimizationStage {
    fn name(&self) -> &str {
        "optimization"
    }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph) -> AphelionResult<()> {
        ctx.trace.info("optimization", "Running graph optimizations");
        // Optimization logic here
        Ok(())
    }
}
```

### Custom Backends

Implement the `Backend` trait for hardware integration:

```rust
use aphelion_core::backend::{Backend, DeviceCapabilities};
use aphelion_core::error::AphelionResult;

struct MyGpuBackend {
    device_id: u32,
}

impl Backend for MyGpuBackend {
    fn name(&self) -> &str { "my_gpu" }
    fn device(&self) -> &str { "cuda:0" }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            supports_f16: true,
            supports_bf16: true,
            supports_tf32: true,
            max_memory_bytes: Some(8 * 1024 * 1024 * 1024),
            compute_units: Some(2560),
        }
    }

    fn is_available(&self) -> bool { true }
    fn initialize(&mut self) -> AphelionResult<()> { Ok(()) }
    fn shutdown(&mut self) -> AphelionResult<()> { Ok(()) }
}
```

### Diagnostics and Tracing

Structured event logging with helper methods:

```rust
use aphelion_core::diagnostics::{InMemoryTraceSink, TraceSinkExt};

let trace = InMemoryTraceSink::new();

// Helper methods reduce boilerplate
trace.info("model.init", "Initializing model");
trace.warn("config.deprecated", "Parameter 'old_name' is deprecated");
trace.error("validation.failed", "Invalid configuration detected");

// Export to JSON
let json = trace.to_json();
```

### Validation

Composable validators for configuration checking:

```rust
use aphelion_core::validation::{CompositeValidator, NameValidator, VersionValidator};

let validator = CompositeValidator::new()
    .with_validator(Box::new(NameValidator))
    .with_validator(Box::new(VersionValidator));

match validator.validate(&config) {
    Ok(_) => println!("Configuration valid"),
    Err(errors) => {
        for error in errors {
            eprintln!("Validation error: {}", error);
        }
    }
}
```

### Async Execution

With the `tokio` feature enabled:

```rust
#[tokio::main]
async fn main() -> AphelionResult<()> {
    let pipeline = BuildPipeline::new()
        .with_async_stage(Box::new(ValidationStage))
        .with_async_stage(Box::new(HashingStage));

    let result = pipeline.execute_async(&ctx, graph).await?;
    Ok(())
}
```

## Error Handling

All operations return `AphelionResult<T>`. Errors include context for debugging:

```rust
use aphelion_core::error::AphelionError;

// Create errors with context
let error = AphelionError::validation("Invalid parameter value")
    .in_stage("config_parsing")
    .for_field("hidden_size");

// Errors chain via std::error::Error::source()
if let Some(source) = error.source() {
    eprintln!("Caused by: {}", source);
}
```

## Project Structure

```
aphelion-framework-rs/
├── crates/
│   ├── aphelion-core/      # Core library: graphs, pipelines, backends
│   ├── aphelion-macros/    # Proc macros: #[aphelion_model]
│   ├── aphelion-python/    # Python bindings via PyO3
│   ├── aphelion-tests/     # Integration tests
│   └── aphelion-examples/  # Usage examples
├── docs/
│   ├── ARCHITECTURE.md     # System design and data flow
│   └── API-GUIDE.md        # Usage patterns with examples
└── SPEC.md                 # Success criteria and compliance
```

## Testing

```bash
# Run all tests
cargo test --workspace

# With async support
cargo test --workspace --features tokio

# With all features
cargo test --workspace --all-features
```

Test coverage: 252 tests across unit, integration, and documentation tests.

## Design Principles

**Determinism**: Graph hashes are reproducible. BTreeMap ensures ordered iteration. No implicit randomness.

**Explicit over implicit**: No hidden state. Configuration is explicit. Errors are typed and contextual.

**Composition over inheritance**: Pipelines compose stages. Validators compose validators. Backends implement traits.

**Zero-cost abstractions**: Feature flags gate optional dependencies. Unused code is not compiled.

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, module responsibilities, data flow |
| [API-GUIDE.md](docs/API-GUIDE.md) | Common patterns with working examples |
| [SPEC.md](SPEC.md) | Success criteria and compliance checklist |

## Examples

Run examples from the workspace:

```bash
cargo run --package aphelion-examples --example basic_usage
cargo run --package aphelion-examples --example custom_backend
cargo run --package aphelion-examples --example pipeline_stages
cargo run --package aphelion-examples --example validation
```

## Version History

| Version | Features |
|---------|----------|
| 1.2.1 | rust-ai-core v0.2.6 integration, Python bindings, memory tracking |
| 1.1.0 | Typed params, preset pipelines, async execution, backend auto-detect |
| 1.0.0 | Core pipeline, graph, config, validation, tracing |

## Contributing

1. Fork the repository
2. Create a feature branch from `develop`
3. Write tests for new functionality
4. Ensure `cargo test --workspace --all-features` passes
5. Ensure `cargo clippy --workspace --all-features -- -D warnings` passes
6. Submit a pull request to `develop`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
