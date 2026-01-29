# Aphelion Framework

[![CI](https://github.com/tzervas/aphelion-framework-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/tzervas/aphelion-framework-rs/actions/workflows/ci.yml)
[![Security Audit](https://github.com/tzervas/aphelion-framework-rs/actions/workflows/security.yml/badge.svg)](https://github.com/tzervas/aphelion-framework-rs/actions/workflows/security.yml)
[![crates.io](https://img.shields.io/crates/v/aphelion-core.svg)](https://crates.io/crates/aphelion-core)
[![docs.rs](https://docs.rs/aphelion-core/badge.svg)](https://docs.rs/aphelion-core)
[![PyPI](https://img.shields.io/pypi/v/aphelion-framework.svg)](https://pypi.org/project/aphelion-framework/)
[![npm](https://img.shields.io/npm/v/aphelion-framework.svg)](https://www.npmjs.com/package/aphelion-framework)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![MSRV](https://img.shields.io/badge/MSRV-1.92-blue.svg)](https://blog.rust-lang.org/)

A unified frontend for AI model development in Rust.

> **Security Note**: This project uses maintained forks (`qlora-paste`, `qlora-gemm`, `qlora-candle`) to replace unmaintained transitive dependencies. See [SECURITY.md](SECURITY.md#unmaintained-dependency-mitigation) for details. Upstream PR: [huggingface/candle#3335](https://github.com/huggingface/candle/pull/3335)

## Overview

Aphelion is a framework frontend that provides an easier entrypoint for AI engineering tasks. It unifies access to Rust AI libraries through a consistent API, handling the complexity of model configuration, pipeline orchestration, and hardware abstraction so you can focus on building models.

### Quick Demo

```
═══════════════════════════════════════════════════════════════
                 Aphelion Framework Demo
═══════════════════════════════════════════════════════════════

📋 Model Configuration:
   Name: transformer
   Version: 1.0.0
   d_model = 512, n_heads = 8, n_layers = 6

🔷 Build Graph:
   Nodes: 2
   Edges: 1
   Hash: c42707e03af1b7e4baf61a94150b1541...

⚙️  Backend: null (cpu)

✅ Pipeline Execution Complete
   Final Hash: c42707e03af1b7e4baf61a94150b1541...
   Trace Events: 2

═══════════════════════════════════════════════════════════════
```

See [docs/screenshots/](docs/screenshots/) for more output examples including ternary acceleration.

### Why Aphelion

Building AI systems in Rust means integrating multiple libraries: tensor operations, memory management, device handling, training loops. Aphelion provides:

- **Unified API**: One consistent interface to underlying AI libraries
- **Reusable Components**: Configurations and pipelines can be templated and shared
- **Deterministic Builds**: SHA-256 hashing ensures identical configs produce identical models
- **Configuration Management**: Type-safe parameters with validation and versioning
- **Pipeline Orchestration**: Composable stages for training, inference, and deployment
- **Backend Abstraction**: Write once, run on CPU, GPU, or accelerators
- **Diagnostics**: Structured tracing for debugging and observability

---

## Ecosystem & Component Crates

Aphelion is built on the **rust-ai ecosystem**, a collection of crates providing CUDA-first GPU acceleration, BitNet b1.58 ternary quantization, and VSA-based gradient compression. These crates work together to enable efficient training and inference.

### Core Foundation: rust-ai-core

[**rust-ai-core**](https://github.com/tzervas/rust-ai-core) (v0.3.1) is the shared foundation layer providing:

- **CUDA-first device selection**: Automatic GPU detection with environment variable overrides
- **Memory tracking**: Budget allocation and peak usage monitoring for GPU memory
- **DType utilities**: Bytes per element, floating-point detection, accumulator types
- **CubeCL interop**: Seamless Candle ↔ CubeCL tensor conversion
- **Common traits**: `ValidatableConfig`, `Quantize`, `Dequantize`, `GpuDispatchable`
- **Unified errors**: `CoreError` hierarchy shared across all rust-ai crates

```bash
# Rust
cargo add rust-ai-core

# Python
pip install rust-ai-core-bindings
```

### Ternary Acceleration: tritter-accel

[**tritter-accel**](https://github.com/tzervas/rust-ai) (v0.1.3) provides BitNet b1.58 ternary operations and VSA gradient compression:

- **Ternary weight packing**: 2-bit per trit storage (4x memory reduction from f32)
- **Ternary matmul**: Addition-only arithmetic (2-4x speedup)
- **AbsMean quantization**: Convert float weights to ternary {-1, 0, +1}
- **VSA gradient compression**: 10-100x compression with <5% accuracy loss

```python
from tritter_accel import (
    pack_ternary_weights,
    ternary_matmul,
    quantize_weights_absmean,
    compress_gradients_vsa,
    decompress_gradients_vsa,
)

# Quantize to ternary and run efficient matmul
ternary_weights, scales = quantize_weights_absmean(float_weights)
packed, scales = pack_ternary_weights(ternary_weights, scales)
output = ternary_matmul(input, packed, scales, original_shape)

# VSA gradient compression for distributed training
compressed = compress_gradients_vsa(gradients, compression_ratio=0.1)
recovered = decompress_gradients_vsa(compressed, original_shape)
```

### Sister Crates

| Crate | Version | Purpose | Repository |
|-------|---------|---------|------------|
| [bitnet-quantize](https://crates.io/crates/bitnet-quantize) | 0.1.1 | Microsoft BitNet b1.58 quantization | [rust-ai](https://github.com/tzervas/rust-ai) |
| [trit-vsa](https://crates.io/crates/trit-vsa) | 0.1.1 | Balanced ternary arithmetic & VSA ops | [rust-ai](https://github.com/tzervas/rust-ai) |
| [vsa-optim-rs](https://crates.io/crates/vsa-optim-rs) | 0.1.1 | Deterministic training optimization | [rust-ai](https://github.com/tzervas/rust-ai) |
| [peft-rs](https://crates.io/crates/peft-rs) | 1.0.1 | LoRA, DoRA, AdaLoRA adapters | [rust-ai](https://github.com/tzervas/rust-ai) |
| [axolotl-rs](https://crates.io/crates/axolotl-rs) | 1.1.1 | YAML-driven LLM fine-tuning | [rust-ai](https://github.com/tzervas/rust-ai) |

### Ecosystem Diagram

```
                           ┌─────────────────────────────────────┐
                           │         Aphelion Framework          │
                           │   Unified API, Pipelines, Configs   │
                           └─────────────────┬───────────────────┘
                                             │
        ┌────────────────┬───────────────────┼───────────────────┬────────────────┐
        │                │                   │                   │                │
        ▼                ▼                   ▼                   ▼                ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ rust-ai-core  │ │tritter-accel  │ │bitnet-quantize│ │   peft-rs     │ │  axolotl-rs   │
│ Device/Memory │ │ Ternary Ops   │ │  B1.58 Quant  │ │ LoRA/DoRA     │ │ Fine-tuning   │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └───────────────┘ └───────────────┘
        │                 │                 │
        └────────┬────────┴────────┬────────┘
                 │                 │
        ┌────────▼────────┐ ┌──────▼───────┐
        │    trit-vsa     │ │  vsa-optim   │
        │Ternary Arithmetic│ │  VSA Optim   │
        └─────────────────┘ └──────────────┘
```

---

## Usage Guide

### Standard Inference & Training

For conventional deep learning workflows using Candle tensors:

```rust
use aphelion_core::prelude::*;

// Build model configuration
let config = ModelConfig::new("transformer", "1.0.0")
    .with_param("d_model", serde_json::json!(512))
    .with_param("n_heads", serde_json::json!(8))
    .with_param("n_layers", serde_json::json!(6));

// Create build graph
let mut graph = BuildGraph::default();
let encoder = graph.add_node("encoder", config);

// Set up backend with device detection
let backend = NullBackend::gpu(0);  // Or use rust-ai-core device detection
let trace = InMemoryTraceSink::new();
let ctx = BuildContext::new(&backend, &trace);

// Execute standard pipeline
let pipeline = BuildPipeline::for_training();
let result = pipeline.execute(&ctx, graph)?;

println!("Model hash: {}", result.stable_hash());
```

With rust-ai-core for memory-aware execution:

```rust
#[cfg(feature = "rust-ai-core")]
{
    use aphelion_core::rust_ai_core::{MemoryTracker, get_device, warn_if_cpu};
    
    // CUDA-first device selection
    let device = get_device(&Default::default())?;
    warn_if_cpu(&device, "aphelion");
    
    // Memory tracking for GPU budget management
    let tracker = MemoryTracker::with_limit(8 * 1024 * 1024 * 1024);  // 8 GB
    tracker.allocate(estimated_model_bytes)?;
    
    println!("Memory: {} / {} bytes", tracker.allocated(), tracker.limit());
}
```

### BitNet B1.58 Ternary Inference

For memory-efficient inference using ternary quantization (16x memory reduction):

```rust
#[cfg(feature = "tritter-accel")]
use aphelion_core::acceleration::{AccelerationStage, InferenceAccelConfig};

// Configure ternary inference
let config = InferenceAccelConfig::new(32)  // batch size 32
    .with_kv_cache(2048);  // max sequence length

// Build accelerated pipeline
let pipeline = BuildPipeline::new()
    .with_stage(Box::new(ValidationStage))
    .with_stage(Box::new(AccelerationStage::with_inference_config(config)))
    .with_stage(Box::new(HashingStage));

let result = pipeline.execute(&ctx, graph)?;

// Metadata now contains acceleration hints:
// - accel.mode = "inference"
// - accel.ternary_layers = true
// - accel.kv_cache = true
// - accel.batch_size = 32
```

Python with tritter-accel:

```python
import numpy as np
from tritter_accel import (
    quantize_weights_absmean,
    pack_ternary_weights,
    ternary_matmul,
)

# Load your model weights
weights = np.random.randn(4096, 4096).astype(np.float32)
inputs = np.random.randn(1, 4096).astype(np.float32)

# Quantize to ternary (16x memory reduction)
ternary_weights, scales = quantize_weights_absmean(weights)
packed, scales = pack_ternary_weights(ternary_weights, scales)

# Efficient ternary matmul (2-4x speedup)
output = ternary_matmul(inputs, packed, scales, weights.shape)
```

### VSA Gradient Compression Training

For distributed training with 10-100x gradient compression:

```rust
#[cfg(feature = "tritter-accel")]
use aphelion_core::acceleration::{AccelerationStage, TrainingAccelConfig};

// Configure VSA-compressed training
let config = TrainingAccelConfig::new(0.1)  // 10x compression
    .with_seed(42)        // deterministic
    .with_mixed_precision();

// Build training pipeline
let pipeline = BuildPipeline::new()
    .with_stage(Box::new(ValidationStage))
    .with_stage(Box::new(AccelerationStage::with_training_config(config)))
    .with_stage(Box::new(HashingStage));

let result = pipeline.execute(&ctx, graph)?;

// Metadata now contains:
// - accel.mode = "training"
// - accel.compression_ratio = 0.1
// - accel.deterministic = true
// - accel.seed = 42
// - accel.mixed_precision = true
```

Python gradient compression:

```python
import numpy as np
from tritter_accel import compress_gradients_vsa, decompress_gradients_vsa

# Your training gradients
gradients = np.random.randn(1_000_000).astype(np.float32)

# Compress for distributed communication (10x reduction)
compressed = compress_gradients_vsa(gradients, compression_ratio=0.1, seed=42)

# Decompress on receiving node
recovered = decompress_gradients_vsa(compressed, gradients.shape, seed=42)

# ~5% accuracy loss, 10x bandwidth reduction
print(f"Original: {gradients.nbytes} bytes")
print(f"Compressed: {len(compressed)} bytes")
```

### Predictive Hybrid Training

Combine gradient compression with deterministic phase training for reproducible, efficient training:

```rust
#[cfg(feature = "tritter-accel")]
use aphelion_core::tritter_backend::{TriterAccelBackend, TriterDevice, TrainingConfig};

// Create accelerated backend
let mut backend = TriterAccelBackend::new(TriterDevice::Cuda(0))
    .with_training_mode(TrainingConfig::new(0.1)
        .with_deterministic(true)
        .with_seed(42))
    .expect("Failed to initialize");

backend.initialize()?;

// Build context with accelerated backend
let trace = InMemoryTraceSink::new();
let ctx = BuildContext::new(&backend, &trace);

// The backend now:
// - Uses VSA gradient compression (10x bandwidth reduction)
// - Enables deterministic training for reproducibility
// - Supports closed-form gradient prediction (experimental)
```

---

### Ecosystem Integration

Aphelion integrates with the Rust AI ecosystem, providing a frontend to:

| Library | Purpose | Feature Flag |
|---------|---------|--------------|
| [rust-ai-core](https://crates.io/crates/rust-ai-core) | Memory tracking, device detection, dtype utilities | `rust-ai-core` |
| [tritter-accel](https://crates.io/crates/tritter-accel) | BitNet b1.58, ternary ops, VSA compression | `tritter-accel` |
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
aphelion-core = { version = "1.2", features = ["rust-ai-core", "tritter-accel", "tokio"] }
```

| Feature | Description |
|---------|-------------|
| `rust-ai-core` | Memory tracking, device detection, dtype utilities via [rust-ai-core](https://crates.io/crates/rust-ai-core) |
| `tritter-accel` | BitNet b1.58 ternary ops, VSA gradient compression via [tritter-accel](https://crates.io/crates/tritter-accel) |
| `cuda` | CUDA GPU support (requires `rust-ai-core`) |
| `burn` | Burn deep learning framework backend |
| `cubecl` | CubeCL GPU compute backend |
| `tokio` | Async pipeline execution support |
| `python` | Python bindings via PyO3 |
| `wasm` | WebAssembly/TypeScript bindings via wasm-bindgen |

### Python

```bash
pip install aphelion-framework
```

The package includes all core features. Memory tracking and device detection are available when the wheel is built with `rust-ai-core` feature.

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

### TypeScript / JavaScript

```bash
npm install aphelion-framework
```

The npm package provides WebAssembly bindings that work in both browsers and Node.js environments.

TypeScript/JavaScript usage:

```typescript
import init, {
  ModelConfig,
  BuildGraph,
  BuildPipeline,
  BuildContext,
  getVersion
} from 'aphelion-framework';

// Initialize WASM module
await init();

console.log(`Aphelion v${getVersion()}`);

// Build configuration
const config = new ModelConfig("transformer", "1.0.0");
const configWithParams = config
  .withParam("d_model", 512)
  .withParam("n_heads", 8);

// Build graph
const graph = new BuildGraph();
const nodeId = graph.addNode("encoder", configWithParams);

// Execute pipeline
const ctx = BuildContext.withNullBackend();
const pipeline = BuildPipeline.standard();
const result = pipeline.execute(ctx, graph);

console.log(`Hash: ${result.stableHash()}`);
console.log(`Nodes: ${result.nodeCount()}`);
```

For Node.js CommonJS:

```javascript
const aphelion = require('aphelion-framework');

aphelion.default().then(() => {
  const config = new aphelion.ModelConfig("model", "1.0.0");
  // ... rest of usage
});
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
│   ├── aphelion-core/      # Core library: graphs, pipelines, backends, Python/WASM bindings
│   │   └── src/
│   │       ├── python/     # Python bindings (--features python)
│   │       └── wasm/       # TypeScript/WASM bindings (--features wasm)
│   ├── aphelion-macros/    # Proc macros: #[aphelion_model]
│   ├── aphelion-tests/     # Integration tests
│   └── aphelion-examples/  # Usage examples
├── docs/
│   ├── ARCHITECTURE.md     # System design and data flow
│   └── API-GUIDE.md        # Usage patterns with examples
└── SPEC.md                 # Success criteria and compliance
```

- Python bindings are built from `aphelion-core` with the `python` feature enabled.
- TypeScript/WASM bindings are built from `aphelion-core` with the `wasm` feature using wasm-pack.

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
# Basic framework demo
cargo run --package aphelion-examples --example demo

# Ternary acceleration demo (BitNet b1.58 + VSA compression)
cargo run --package aphelion-examples --example tritter_demo --features tritter-accel
```

Example output from tritter_demo:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔷 BitNet B1.58 Ternary Inference
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Inference Configuration Applied:
   Mode: "inference"
   Ternary Layers: true (16x memory reduction)
   Batch Size: 32
   KV Cache: true

   Memory Savings:
   - Float32 weights: ~28 GB (7B × 4 bytes)
   - Ternary packed:  ~1.75 GB (7B × 2 bits / 8)
   - Reduction: 16x
```

## Version History

| Version | Features |
|---------|----------|
| 1.2.9 | TypeScript/WASM bindings via wasm-bindgen (`wasm` feature), npm package |
| 1.2.8 | Unified Python bindings into aphelion-core (`python` feature), rust-ai-core 0.3.1, candle-core 0.9.2 |
| 1.2.7 | Dependency updates: rust-ai-core 0.3.1, tritter-accel 0.1.3 with pyo3 0.27.2 compatibility |
| 1.2.6 | Fix: Python package now uses dynamic versioning from Cargo.toml |
| 1.2.5 | Dependency updates: burn 0.20.1, pyo3 0.27.2, thiserror 2.0.18, half 2.7.1 |
| 1.2.4 | Security: Replaced unmaintained `paste`/`gemm` with maintained forks. See [SECURITY.md](SECURITY.md#unmaintained-dependency-mitigation) |
| 1.2.3 | Documentation updates for dependency tracking |
| 1.2.2 | tritter-accel integration, enhanced documentation |
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

## Attributions

Aphelion Framework builds on excellent work from the Rust and AI communities:

| Project | License | Use |
|---------|---------|-----|
| [rust-ai-core](https://github.com/tzervas/rust-ai-core) | MIT | Foundation layer: device selection, memory, dtypes |
| [tritter-accel](https://github.com/tzervas/rust-ai) | MIT | BitNet b1.58, ternary ops, VSA compression |
| [bitnet-quantize](https://github.com/tzervas/rust-ai) | MIT | Microsoft BitNet b1.58 quantization |
| [trit-vsa](https://github.com/tzervas/rust-ai) | MIT | Balanced ternary arithmetic |
| [vsa-optim-rs](https://github.com/tzervas/rust-ai) | MIT | VSA gradient optimization |
| [Candle](https://github.com/huggingface/candle) | MIT/Apache-2.0 | Tensor operations (via rust-ai-core) |
| [Burn](https://github.com/tracel-ai/burn) | MIT/Apache-2.0 | Deep learning framework backend |
| [CubeCL](https://github.com/tracel-ai/cubecl) | MIT/Apache-2.0 | GPU compute |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
