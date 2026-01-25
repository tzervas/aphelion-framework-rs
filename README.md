# Aphelion Framework (Rust)

A Rust AI framework focused on fast, traceable, and reproducible model building with deep customization.

## Quick Start

Get up and running in 5 minutes with this minimal example.

### Installation

Add Aphelion to your `Cargo.toml`:

```toml
[dependencies]
aphelion-core = "1.0"
```

For burn backend integration:

```toml
[dependencies]
aphelion-core = { version = "1.0", features = ["burn"] }
```

### Minimal Example

```rust
use aphelion_core::prelude::*;

fn main() -> AphelionResult<()> {
    // 1. Create a model configuration
    let config = ModelConfig::new("my-model", "1.0.0")
        .with_param("hidden_size", serde_json::json!(256));

    // 2. Build a computation graph
    let mut graph = BuildGraph::default();
    graph.add_node("layer", config);

    // 3. Set up backend and diagnostics
    let backend = NullBackend::cpu();
    let trace = InMemoryTraceSink::new();
    let ctx = BuildContext::new(&backend, &trace);

    // 4. Execute pipeline
    let pipeline = BuildPipeline::new();
    let result = pipeline.execute(&ctx, graph)?;

    println!("Graph hash: {}", result.stable_hash());
    Ok(())
}
```

### Feature Flags

- `burn` - Burn deep learning framework integration (recommended)
- `cubecl` - CubeCL GPU compute integration
- `rust-ai-core` - Rust AI Core ecosystem integration

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - System design and data flow
- [API Guide](docs/API-GUIDE.md) - Common patterns and examples
- [Canonical Spec](SPEC.md) - Success criteria and compliance
- [Expanded Spec](docs/spec.md) - Development notes

## Examples

Check the [examples crate](crates/aphelion-examples/src/) for detailed usage patterns:

- `basic_usage.rs` - Simple model building workflow
- `custom_backend.rs` - Implementing custom backends
- `pipeline_stages.rs` - Custom pipeline stages and hooks
- `validation.rs` - Configuration validation patterns

## Status

Version 1.0.0 is complete with all core features:
- [x] SC-1 Deterministic graph hash
- [x] SC-2 Config validation enforced
- [x] SC-3 Trace capture works
- [x] SC-4 Macro ergonomic contract
- [x] SC-5 Burn-first backend availability

Version 1.1.0 in progress with developer experience enhancements.

## License

MIT © 2026 Tyler Zervas (tzervas) <tz-dev@vectorweight.com>
