# Aphelion Framework Architecture

This document provides a high-level overview of the Aphelion framework's architecture, module responsibilities, data flow, and extension points.

## System Overview

Aphelion is designed around the principle of **transparent, traceable model construction** with backend abstraction. The framework enables developers to define AI models once and run them on different hardware backends (CPU, GPU, TPU) while maintaining full traceability of the build process.

```
┌─────────────────────────────────────────────────────────────┐
│                     Aphelion Framework                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐     ┌──────────┐     ┌───────────┐          │
│  │  Config  │────▶│  Graph   │────▶│ Pipeline  │          │
│  │ ModelConf│     │BuildGraph│     │BuildPipe  │          │
│  └──────────┘     └──────────┘     └─────┬─────┘          │
│                                           │                 │
│                                           ▼                 │
│  ┌──────────┐     ┌──────────┐     ┌───────────┐          │
│  │ Backend  │◀────│  Context │◀────│Validation │          │
│  │ Abstrac  │     │BuildCtx  │     │ Rules     │          │
│  └──────────┘     └──────────┘     └───────────┘          │
│        │                                                    │
│        ▼                                                    │
│  ┌──────────┐     ┌──────────┐                            │
│  │Diagnostc │◀────│  Trace   │                            │
│  │TraceSink │     │  Events  │                            │
│  └──────────┘     └──────────┘                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Core Modules

### 1. Configuration (`config`)

**Responsibility**: Model metadata and parameter management

The `config` module provides type-safe model configuration with deterministic ordering:

- `ModelConfig` - Core configuration container with name, version, and parameters
- `ConfigSpec` - Trait for custom configuration types
- `ModelConfigBuilder` - Builder pattern for complex configurations

**Key Design Principles**:
- Uses `BTreeMap` for deterministic parameter ordering
- Supports semantic versioning
- Enables type-safe parameter access with `param::<T>()`
- Serializable to/from JSON

**Determinism Guarantee**: Two configs with identical data will serialize to identical JSON and produce identical hashes.

### 2. Graph (`graph`)

**Responsibility**: DAG representation of model architecture

The `graph` module represents models as directed acyclic graphs:

- `BuildGraph` - Main graph structure with nodes and edges
- `GraphNode` - Individual computation nodes with metadata
- `NodeId` - Type-safe node identifiers
- `stable_hash()` - Deterministic SHA256 hash of graph structure

**Key Design Principles**:
- Nodes store `ModelConfig` and arbitrary metadata
- Edges define dependencies between nodes
- Supports graph traversal and analysis
- Hash is stable across runs for identical structures

**Determinism Guarantee**: Graph hashing uses sorted node IDs and canonical edge ordering to ensure reproducibility.

### 3. Backend (`backend`)

**Responsibility**: Hardware abstraction layer

The `backend` module provides a trait-based abstraction for different computational backends:

- `Backend` - Main trait for backend implementations
- `DeviceCapabilities` - Hardware feature reporting (FP16, BF16, TF32, memory)
- `MemoryInfo` - Runtime memory usage statistics
- `NullBackend` - Testing/prototyping backend
- `BackendRegistry` - Multi-backend management

**Key Design Principles**:
- Backend implementations are pluggable
- Capability reporting enables model optimization
- Burn framework is the primary production backend
- NullBackend for testing without hardware dependencies

**Extension Point**: Implement the `Backend` trait to add support for new hardware or frameworks.

### 4. Pipeline (`pipeline`)

**Responsibility**: Build orchestration and execution

The `pipeline` module coordinates multi-stage build processes:

- `BuildPipeline` - Main orchestrator with stages and hooks
- `PipelineStage` - Trait for custom build stages
- `BuildContext` - Runtime context with backend and trace sink
- Pre/post hooks for custom logic
- Progress tracking callbacks
- Stage skipping support

**Key Design Principles**:
- Stages execute sequentially, modifying the graph
- Composable and reusable stage implementations
- Hooks enable pre/post processing
- Preset pipelines for common workflows (standard, training, inference)

**Extension Point**: Implement `PipelineStage` to add custom build steps (validation, optimization, code generation).

### 5. Validation (`validation`)

**Responsibility**: Configuration and data validation

The `validation` module ensures configurations meet requirements:

- `ConfigValidator` - Trait for validation logic
- `NameValidator` - Model name validation (alphanumeric + hyphens/underscores)
- `VersionValidator` - Semantic version validation (X.Y.Z format)
- `CompositeValidator` - Combines multiple validators
- `ValidationError` - Typed validation errors

**Key Design Principles**:
- Validators are composable and reusable
- Fail-fast with clear error messages
- Built-in validators for common constraints
- Custom validators for domain-specific rules

**Extension Point**: Implement `ConfigValidator` for custom validation rules.

### 6. Diagnostics (`diagnostics`)

**Responsibility**: Tracing and event recording

The `diagnostics` module captures build process events:

- `TraceEvent` - Individual diagnostic events with timestamps
- `TraceSink` - Trait for event storage backends
- `InMemoryTraceSink` - In-memory event collection
- `TraceSinkExt` - Helper methods for common operations
- `TraceLevel` - Severity levels (Debug, Info, Warn, Error)

**Key Design Principles**:
- All events are timestamped
- Thread-safe event recording
- Supports span/trace IDs for distributed tracing
- JSON export for analysis

**Extension Point**: Implement `TraceSink` for custom logging backends (files, databases, telemetry systems).

### 7. Export (`export`)

**Responsibility**: JSON serialization and data export

The `export` module handles data serialization:

- JSON export of trace events
- Graph structure serialization
- Configuration export

**Key Design Principles**:
- Uses `serde` for robust serialization
- Pretty-printing support for human readability
- Deterministic output ordering

## Data Flow

### Basic Model Building Flow

```
1. Create ModelConfig
   ↓
2. Build BuildGraph (add nodes and edges)
   ↓
3. Create Backend and TraceSink
   ↓
4. Create BuildContext with backend and trace
   ↓
5. Create BuildPipeline with stages
   ↓
6. Execute pipeline with context and graph
   ↓
7. Receive final graph with computed hash
   ↓
8. Export trace events for inspection
```

### Pipeline Execution Flow

```
BuildPipeline::execute(ctx, graph)
   ↓
   ├─ Execute pre-hooks (ctx)
   ↓
   ├─ For each stage:
   │    ├─ Check if skipped
   │    ├─ Call progress callback
   │    ├─ Execute stage.execute(ctx, graph)
   │    └─ Record trace events
   ↓
   ├─ Execute post-hooks (ctx, graph)
   ↓
   └─ Return final graph
```

### Trace Event Flow

```
Pipeline/Stage
   ↓
ctx.trace.record(TraceEvent)
   ↓
TraceSink implementation
   ↓
InMemoryTraceSink stores events
   ↓
export::export_traces_to_json()
   ↓
JSON file or string output
```

## Extension Points

### 1. Custom Backends

Implement the `Backend` trait to add support for new hardware or frameworks:

```rust
use aphelion_core::backend::{Backend, DeviceCapabilities};
use aphelion_core::error::AphelionResult;

struct MyCustomBackend {
    device: String,
}

impl Backend for MyCustomBackend {
    fn name(&self) -> &str { "my_backend" }
    fn device(&self) -> &str { &self.device }
    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            supports_f16: true,
            supports_bf16: true,
            supports_tf32: false,
            max_memory_bytes: Some(16 * 1024 * 1024 * 1024),
            compute_units: Some(2048),
        }
    }
    fn is_available(&self) -> bool { true }
    fn initialize(&mut self) -> AphelionResult<()> { Ok(()) }
    fn shutdown(&mut self) -> AphelionResult<()> { Ok(()) }
}
```

### 2. Custom Pipeline Stages

Implement `PipelineStage` for custom build logic:

```rust
use aphelion_core::pipeline::{PipelineStage, BuildContext};
use aphelion_core::graph::BuildGraph;
use aphelion_core::error::AphelionResult;

struct MyOptimizationStage;

impl PipelineStage for MyOptimizationStage {
    fn name(&self) -> &str { "optimization" }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph)
        -> AphelionResult<()>
    {
        // Custom optimization logic
        ctx.trace.info("optimization", "Running optimizations");
        // Modify graph as needed
        Ok(())
    }
}
```

### 3. Custom Validators

Implement `ConfigValidator` for domain-specific validation:

```rust
use aphelion_core::validation::{ConfigValidator, ValidationError};
use aphelion_core::config::ModelConfig;

struct MyDomainValidator;

impl ConfigValidator for MyDomainValidator {
    fn validate(&self, config: &ModelConfig)
        -> Result<(), Vec<ValidationError>>
    {
        let mut errors = Vec::new();

        // Custom validation logic
        if !config.params.contains_key("required_field") {
            errors.push(ValidationError::new(
                "required_field",
                "This field is required"
            ));
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}
```

### 4. Custom Trace Sinks

Implement `TraceSink` for custom logging backends:

```rust
use aphelion_core::diagnostics::{TraceSink, TraceEvent};

struct MyDatabaseTraceSink {
    // Database connection
}

impl TraceSink for MyDatabaseTraceSink {
    fn record(&self, event: TraceEvent) {
        // Write to database
    }
}
```

## Design Invariants

### Determinism

**Invariant**: Given identical inputs, all operations produce identical outputs.

**Implementation**:
- All maps use `BTreeMap` for sorted keys
- Graph hashing uses canonical ordering
- Timestamps are optional and excluded from hashing
- Random number generation requires explicit seed

### Thread Safety

**Invariant**: All public APIs are thread-safe.

**Implementation**:
- Trait bounds require `Send + Sync`
- Interior mutability uses `Arc<Mutex<>>` or atomic types
- TraceSink implementations must be thread-safe

### Error Handling

**Invariant**: No panics in library code; all errors are returned as `Result<T, AphelionError>`.

**Implementation**:
- `AphelionError` enum covers all error cases
- Errors include context (field, stage, source chain)
- Uses `thiserror` for ergonomic error types
- No `unwrap()` or `expect()` in library code

### Backward Compatibility

**Invariant**: Minor version updates preserve API compatibility.

**Implementation**:
- New fields added with `#[serde(default)]`
- Deprecated APIs marked with `#[deprecated]`
- Breaking changes only in major versions

## Performance Considerations

### Memory Efficiency

- Graphs use compact node/edge representation
- Trace events can be filtered by level to reduce memory
- Backend implementations can report memory constraints
- Large graphs should be built incrementally

### Computational Efficiency

- Graph hashing is cached when possible
- Validation runs only when enabled
- Pipeline stages can be skipped conditionally
- Backend capabilities enable hardware-specific optimizations

### Concurrency

- Pipeline stages execute sequentially (by design for determinism)
- Multiple pipelines can execute in parallel
- Trace sinks must handle concurrent event recording
- Backend operations can be parallelized internally

## Testing Strategy

### Unit Tests

- Each module has inline unit tests
- Test determinism, error handling, edge cases
- Use `NullBackend` to avoid hardware dependencies

### Integration Tests

- Located in `aphelion-tests` crate
- Test end-to-end workflows
- Validate success criteria (SC-1 through SC-5)

### Example Tests

- All examples in `aphelion-examples` have test coverage
- Examples serve as documentation and regression tests
- Run with `cargo test --workspace`

## Migration Guide

### From 0.x to 1.0

Version 1.0 is the initial stable release. No migration needed.

### Future Versions

Breaking changes will be documented in CHANGELOG.md with migration examples.
