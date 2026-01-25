# aphelion-python

Python bindings for the [Aphelion AI Framework](https://github.com/tzervas/aphelion-framework-rs).

## Installation

```bash
pip install aphelion
```

## Building from source

Requires [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin
cd crates/aphelion-python
maturin develop
```

### Feature flags

Build with specific backends:

```bash
maturin develop --features burn
maturin develop --features cubecl
maturin develop --features tritter-accel
maturin develop --features full  # all backends
```

## Quick Start

```python
import aphelion
import asyncio

async def main():
    # Create configuration
    config = aphelion.ModelConfig("my-model", "1.0.0")
    config.with_param("hidden_size", 256)
    config.with_param("layers", 4)

    # Build graph
    graph = aphelion.BuildGraph()
    node1 = graph.add_node("input", config)
    node2 = graph.add_node("output", config)
    graph.add_edge(node1, node2)

    # Validate
    assert not graph.has_cycle()
    print(f"Hash: {graph.stable_hash()}")

    # Create pipeline context
    backend = aphelion.NullBackend.cpu()
    trace = aphelion.InMemoryTraceSink()
    ctx = aphelion.BuildContext(backend, trace)

    # Execute pipeline (async)
    pipeline = aphelion.BuildPipeline.standard()
    result = await pipeline.execute_async(ctx, graph)

    # Get trace events
    for event in trace.events():
        print(f"[{event.level}] {event.id}: {event.message}")

asyncio.run(main())
```

## API Overview

### Configuration

- `ModelConfig` - Model configuration with typed parameters
- `ModelConfigBuilder` - Builder pattern for configs

### Graph

- `NodeId` - Unique node identifier
- `GraphNode` - Graph node with config and metadata
- `BuildGraph` - Computation graph with cycle detection and hashing

### Backend

- `NullBackend` - CPU-only reference backend for testing

> **Note:** Additional backends (Burn, CubeCL, Tritter-Accel) will be available
> in future releases. Check `HAS_BURN`, `HAS_CUBECL`, `HAS_TRITTER_ACCEL`
> feature flags at runtime.

### Pipeline

- `BuildPipeline` - Multi-stage build pipeline with hooks
- `BuildContext` - Execution context with backend and tracing
- `PipelineStage` - Custom stage protocol (subclass in Python)

### Diagnostics

- `TraceLevel` - Log levels (Debug, Info, Warn, Error)
- `TraceEvent` - Structured trace event
- `InMemoryTraceSink` - Collects events in memory

### Validation

- `ValidationError` - Validation failure details
- `NameValidator` - Validates model names
- `VersionValidator` - Validates semver versions
- `CompositeValidator` - Combines multiple validators

### Feature Flags

Python constants to check available backends at runtime:

- `HAS_BURN` - Whether Burn backend is available
- `HAS_CUBECL` - Whether CubeCL backend is available
- `HAS_RUST_AI_CORE` - Whether rust-ai-core is available
- `HAS_TRITTER_ACCEL` - Whether Tritter hardware acceleration is available

## License

MIT OR Apache-2.0
