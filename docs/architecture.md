# Aphelion Framework Architecture

This document provides a comprehensive overview of the Aphelion Framework's architecture,
including system design, module dependencies, core concepts, and extension points.

## Table of Contents

1. [High-Level System Architecture](#high-level-system-architecture)
2. [Module Dependency Graph](#module-dependency-graph)
3. [Core Concepts](#core-concepts)
   - [ModelConfig and ConfigSpec](#modelconfig-and-configspec)
   - [BuildGraph and Deterministic Hashing](#buildgraph-and-deterministic-hashing)
   - [Backend Trait and Implementations](#backend-trait-and-implementations)
   - [Pipeline Stages and Hooks](#pipeline-stages-and-hooks)
   - [Trace Events and Sinks](#trace-events-and-sinks)
4. [Data Flow for a Typical Model Build](#data-flow-for-a-typical-model-build)
5. [Extension Points](#extension-points)
6. [Feature Flags Explained](#feature-flags-explained)

---

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APHELION FRAMEWORK                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │   User Code     │    │  aphelion-macros│    │     aphelion-core       │  │
│  │                 │    │                 │    │                         │  │
│  │ #[aphelion_model]    │  #[aphelion_    │    │  ┌─────────────────┐   │  │
│  │ struct MyModel  │───▶│   model]        │───▶│  │   ModelBuilder  │   │  │
│  │ { config, ... } │    │  proc-macro     │    │  │   ConfigSpec    │   │  │
│  │                 │    │                 │    │  └────────┬────────┘   │  │
│  └─────────────────┘    └─────────────────┘    │           │            │  │
│                                                │           ▼            │  │
│  ┌─────────────────────────────────────────────┼───────────────────┐    │  │
│  │                  BUILD PIPELINE             │                   │    │  │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐ │                   │    │  │
│  │  │Validation│──▶│  Build   │──▶│  Export  │ │                   │    │  │
│  │  │  Stage   │   │  Stage   │   │  Stage   │ │                   │    │  │
│  │  └──────────┘   └────┬─────┘   └──────────┘ │                   │    │  │
│  │                      │                      │                   │    │  │
│  └──────────────────────┼──────────────────────┘                   │    │  │
│                         │                                          │    │  │
│           ┌─────────────┼─────────────┐                            │    │  │
│           │             │             │                            │    │  │
│           ▼             ▼             ▼                            │    │  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                   │    │  │
│  │   Backend   │ │ BuildGraph  │ │  TraceSink  │                   │    │  │
│  │  (compute)  │ │  (output)   │ │ (diagnostics)│                  │    │  │
│  └──────┬──────┘ └─────────────┘ └──────┬──────┘                   │    │  │
│         │                               │                          │    │  │
│         ▼                               ▼                          │    │  │
│  ┌────────────────────────────────────────────────────────────┐    │    │  │
│  │                    BACKEND IMPLEMENTATIONS                  │    │    │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │    │    │  │
│  │  │   Null   │  │   Burn   │  │  CubeCL  │  │  Tritter │    │    │    │  │
│  │  │ Backend  │  │ Backend  │  │ Backend  │  │ Backend  │    │    │    │  │
│  │  │ (test)   │  │ (DL)     │  │ (GPU)    │  │ (accel)  │    │    │    │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │    │    │  │
│  │     always       feature:      feature:      feature:       │    │    │  │
│  │    available       burn         cubecl    tritter-accel     │    │    │  │
│  └────────────────────────────────────────────────────────────┘    │    │  │
│                                                                    │    │  │
│  ┌────────────────────────────────────────────────────────────┐    │    │  │
│  │                      DIAGNOSTICS                            │    │    │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │    │    │  │
│  │  │ InMemory     │  │ JsonExporter │  │ Custom TraceSink │  │    │    │  │
│  │  │ TraceSink    │  │              │  │ (user-defined)   │  │    │    │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │    │    │  │
│  └────────────────────────────────────────────────────────────┘    │    │  │
│                                                                    │    │  │
└────────────────────────────────────────────────────────────────────┴────┘  │
                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
                          ┌──────────────────────┐
                          │   aphelion-macros    │
                          │   (proc-macro crate) │
                          └──────────┬───────────┘
                                     │
                                     │ depends on
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                            aphelion-core                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐           │
│  │    error     │◀──────│    config    │◀──────│  validation  │           │
│  │              │       │              │       │              │           │
│  │ AphelionError│       │ ModelConfig  │       │ ConfigValid- │           │
│  │ AphelionResult       │ ConfigSpec   │       │    ator      │           │
│  └──────────────┘       └──────┬───────┘       └──────────────┘           │
│         ▲                      │                      ▲                    │
│         │                      │                      │                    │
│         │               ┌──────▼───────┐              │                    │
│         │               │    graph     │              │                    │
│         │               │              │              │                    │
│         │               │ BuildGraph   │              │                    │
│         │               │ GraphNode    │              │                    │
│         │               │ NodeId       │              │                    │
│         │               │ stable_hash()│              │                    │
│         │               └──────┬───────┘              │                    │
│         │                      │                      │                    │
│  ┌──────┴───────┐       ┌──────▼───────┐       ┌──────┴───────┐           │
│  │  diagnostics │◀──────│   backend    │       │   pipeline   │           │
│  │              │       │              │       │              │           │
│  │ TraceEvent   │       │ Backend trait│──────▶│ BuildPipeline│           │
│  │ TraceSink    │       │ ModelBuilder │       │ BuildContext │           │
│  │ InMemory-    │       │ NullBackend  │       │              │           │
│  │  TraceSink   │       └──────────────┘       └──────────────┘           │
│  └──────┬───────┘              ▲                                          │
│         │                      │                                          │
│         │              ┌───────┴────────────────────┬───────────────┐     │
│         │              │                            │               │     │
│         ▼       ┌──────┴───────┐       ┌───────────┴──┐   ┌────────┴───┐ │
│  ┌─────────────┐│ burn_backend │       │cubecl_backend│   │tritter_    │ │
│  │   export    ││              │       │              │   │ backend    │ │
│  │             ││ BurnBackend  │       │CubeclBackend │   │            │ │
│  │JsonExporter ││ BurnDevice   │       │CubeclDevice  │   │(feature:   │ │
│  │Serializable-││              │       │              │   │ tritter-   │ │
│  │ TraceEvent  ││(feature:burn)│       │(feature:     │   │   accel)   │ │
│  └─────────────┘└──────────────┘       │  cubecl)     │   └────────────┘ │
│                                        └──────────────┘                   │
│                                                                            │
│  ┌─────────────┐       ┌──────────────┐                                   │
│  │   prelude   │       │ rust_ai_core │                                   │
│  │             │       │              │                                   │
│  │ (re-exports │       │ (future      │                                   │
│  │  common     │       │  rust-ai-core│                                   │
│  │  types)     │       │  integration)│                                   │
│  └─────────────┘       └──────────────┘                                   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

                          External Dependencies
                ┌────────────────────────────────────────┐
                │  serde, serde_json  │  Serialization   │
                │  thiserror          │  Error handling  │
                │  tracing            │  Logging         │
                │  sha2, hex          │  Hashing         │
                │  tokio (optional)   │  Async runtime   │
                └────────────────────────────────────────┘
```

---

## Core Concepts

### ModelConfig and ConfigSpec

`ModelConfig` is the foundation for all model configurations in Aphelion. It provides
a deterministic, serializable container for model metadata and parameters.

```rust
/// Generic model configuration container with deterministic ordering.
pub struct ModelConfig {
    pub name: String,              // Model identifier
    pub version: String,           // Semantic version string
    pub params: BTreeMap<String, serde_json::Value>,  // Arbitrary parameters
}
```

**Key Design Decisions:**

1. **BTreeMap for Parameters**: Uses `BTreeMap` instead of `HashMap` to ensure
   deterministic iteration order, critical for reproducible hashing.

2. **JSON Values**: Parameters use `serde_json::Value` for flexibility while
   maintaining serializability.

3. **Builder Pattern**: Fluent API with `with_param()` for ergonomic construction.

**ConfigSpec Trait:**

```rust
/// Trait for types that expose their configuration.
pub trait ConfigSpec: Send + Sync {
    fn config(&self) -> &ModelConfig;
}
```

The `#[aphelion_model]` macro automatically implements this trait for annotated structs.

---

### BuildGraph and Deterministic Hashing

`BuildGraph` represents the computation graph produced by model building.

```
┌─────────────────────────────────────────────────────────────────┐
│                         BuildGraph                              │
├─────────────────────────────────────────────────────────────────┤
│  nodes: Vec<GraphNode>                                          │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │  GraphNode { id: NodeId(1), name: "input", config }     │  │
│    │  GraphNode { id: NodeId(2), name: "hidden", config }    │  │
│    │  GraphNode { id: NodeId(3), name: "output", config }    │  │
│    └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  edges: Vec<(NodeId, NodeId)>                                   │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │  (NodeId(1), NodeId(2))  // input -> hidden             │  │
│    │  (NodeId(2), NodeId(3))  // hidden -> output            │  │
│    └─────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  stable_hash() -> String (SHA-256 hex)                          │
│                                                                 │
│  Hashing includes:                                              │
│  • Node IDs (u64, little-endian bytes)                          │
│  • Node names (UTF-8 bytes)                                     │
│  • Node configs (JSON serialized)                               │
│  • Edge pairs (from/to as u64 bytes)                            │
└─────────────────────────────────────────────────────────────────┘
```

**Deterministic Hashing:**

The `stable_hash()` method produces a cryptographic hash that is:

- **Reproducible**: Same inputs always produce the same hash
- **Collision-resistant**: Different graphs produce different hashes
- **Content-based**: Hash depends on actual graph content, not memory addresses

```rust
pub fn stable_hash(&self) -> String {
    let mut hasher = Sha256::new();
    
    // Hash nodes in order
    for node in &self.nodes {
        hasher.update(node.id.value().to_le_bytes());
        hasher.update(node.name.as_bytes());
        hasher.update(serde_json::to_vec(&node.config).unwrap_or_default());
    }
    
    // Hash edges in order
    for (from, to) in &self.edges {
        hasher.update(from.value().to_le_bytes());
        hasher.update(to.value().to_le_bytes());
    }
    
    hex::encode(hasher.finalize())
}
```

---

### Backend Trait and Implementations

The `Backend` trait abstracts over different execution environments.

```
                      ┌──────────────────────┐
                      │   Backend (trait)    │
                      ├──────────────────────┤
                      │ + name() -> &str     │
                      │ + device() -> &str   │
                      └──────────┬───────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   NullBackend   │   │   BurnBackend   │   │  CubeclBackend  │
├─────────────────┤   ├─────────────────┤   ├─────────────────┤
│ name: "null"    │   │ name: "burn"    │   │ name: "cubecl"  │
│ device: "cpu"   │   │ device: varies  │   │ device: varies  │
│                 │   │                 │   │                 │
│ Always available│   │ GPU/CPU         │   │ GPU compute     │
│ For testing     │   │ Deep learning   │   │ Low-level ops   │
└─────────────────┘   └─────────────────┘   └─────────────────┘
                             │                      │
                             ▼                      ▼
                      ┌─────────────────────────────────────┐
                      │         Device Types                │
                      ├─────────────────────────────────────┤
                      │ • Cpu      - Always available       │
                      │ • Cuda(n)  - NVIDIA GPU index n     │
                      │ • Metal(n) - Apple GPU index n      │
                      │ • Vulkan(n)- Cross-platform GPU     │
                      │ • Wgpu(n)  - WebGPU (CubeCL only)   │
                      └─────────────────────────────────────┘
```

**ModelBuilder Trait:**

```rust
pub trait ModelBuilder: Send + Sync {
    type Output;
    fn build(&self, backend: &dyn Backend, trace: &dyn TraceSink) -> Self::Output;
}
```

The `#[aphelion_model]` macro generates this impl, delegating to a user-defined
`build_graph()` method.

---

### Pipeline Stages and Hooks

The `BuildPipeline` orchestrates the build process through stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            BuildPipeline                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BuildContext { backend, trace }                                            │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    VALIDATION STAGE (optional)                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  • Check config.name is not empty                              │ │   │
│  │  │  • Check config.version is not empty                           │ │   │
│  │  │  • Emit TraceEvent: "pipeline.validate"                        │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │  Return: Err(AphelionError::InvalidConfig) if validation fails      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        BUILD STAGE                                   │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  • Emit TraceEvent: "pipeline.start"                           │ │   │
│  │  │  • Call model.build(backend, trace)                            │ │   │
│  │  │  • Compute graph.stable_hash()                                 │ │   │
│  │  │  • Emit TraceEvent: "pipeline.finish" with hash                │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │  Return: Ok(BuildGraph)                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│  Result<BuildGraph, AphelionError>                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Pipeline Methods:**

| Method | Validation | Use Case |
|--------|------------|----------|
| `build()` | No | Quick builds, testing |
| `build_with_validation()` | Yes | Production builds |

---

### Trace Events and Sinks

The diagnostics system enables observability throughout the build process.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TraceEvent                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  pub struct TraceEvent {                                                    │
│      id: String,           // Event identifier (e.g., "pipeline.start")    │
│      message: String,      // Human-readable description                    │
│      timestamp: SystemTime,// When the event occurred                       │
│      level: TraceLevel,    // Debug | Info | Warn | Error                   │
│      span_id: Option<String>,   // Distributed tracing span                 │
│      trace_id: Option<String>,  // Distributed tracing trace                │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘

                      ┌──────────────────────┐
                      │  TraceSink (trait)   │
                      ├──────────────────────┤
                      │ + record(TraceEvent) │
                      └──────────┬───────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│InMemoryTraceSink│   │  JsonExporter   │   │ Custom Sink     │
├─────────────────┤   ├─────────────────┤   ├─────────────────┤
│ Stores events   │   │ Exports to JSON │   │ User-defined    │
│ in Vec<>        │   │ for analysis    │   │ implementations │
│                 │   │                 │   │                 │
│ events() ->     │   │ to_json() ->    │   │ Log to file,    │
│  Vec<TraceEvent>│   │  String         │   │ send to service │
└─────────────────┘   │ write_to(W)     │   │ etc.            │
                      └─────────────────┘   └─────────────────┘
```

**Event Flow:**

```
Pipeline        TraceSink        JsonExporter        External
   │                │                 │                 │
   │ record(start)  │                 │                 │
   │───────────────▶│                 │                 │
   │                │ store event     │                 │
   │                │                 │                 │
   │   ... build ...                  │                 │
   │                │                 │                 │
   │ record(finish) │                 │                 │
   │───────────────▶│                 │                 │
   │                │ store event     │                 │
   │                │                 │                 │
   │                │    to_json()    │                 │
   │                │────────────────▶│                 │
   │                │                 │  write to file  │
   │                │                 │────────────────▶│
   │                │                 │                 │
```

---

## Data Flow for a Typical Model Build

This section traces a complete build from user code to output.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. USER DEFINES MODEL                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  #[aphelion_model]                                                          │
│  struct MyModel {                                                           │
│      config: ModelConfig,                                                   │
│      layer_sizes: Vec<usize>,                                               │
│  }                                                                          │
│                                                                             │
│  impl MyModel {                                                             │
│      fn build_graph(&self, backend: &dyn Backend,                           │
│                     trace: &dyn TraceSink) -> BuildGraph {                  │
│          // User implementation                                             │
│      }                                                                      │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. MACRO EXPANSION (compile time)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  // Generated by #[aphelion_model]                                          │
│  impl ConfigSpec for MyModel {                                              │
│      fn config(&self) -> &ModelConfig { &self.config }                      │
│  }                                                                          │
│                                                                             │
│  impl ModelBuilder for MyModel {                                            │
│      type Output = BuildGraph;                                              │
│      fn build(&self, backend: &dyn Backend,                                 │
│               trace: &dyn TraceSink) -> BuildGraph {                        │
│          self.build_graph(backend, trace)                                   │
│      }                                                                      │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. RUNTIME: CREATE INSTANCES                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  let model = MyModel {                                                      │
│      config: ModelConfig::new("my-model", "1.0.0")                          │
│          .with_param("hidden_size", json!(256)),                            │
│      layer_sizes: vec![784, 256, 10],                                       │
│  };                                                                         │
│                                                                             │
│  let backend = NullBackend::cpu();                                          │
│  let trace = InMemoryTraceSink::new();                                      │
│  let ctx = BuildContext { backend: &backend, trace: &trace };               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. PIPELINE EXECUTION                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BuildPipeline::build_with_validation(&model, ctx)                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 4a: Validation                                                 │   │
│  │  • config.name = "my-model" ✓                                        │   │
│  │  • config.version = "1.0.0" ✓                                        │   │
│  │  → TraceEvent { id: "pipeline.validate", ... }                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 4b: Build Start                                                │   │
│  │  → TraceEvent { id: "pipeline.start", ... }                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 4c: Model Build                                                │   │
│  │  • model.build(backend, trace) called                                │   │
│  │  • User's build_graph() executes                                     │   │
│  │  • Nodes and edges added to BuildGraph                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Step 4d: Finalization                                               │   │
│  │  • graph.stable_hash() computed                                      │   │
│  │  → TraceEvent { id: "pipeline.finish", hash: "a1b2c3...", ... }      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. OUTPUT                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Result<BuildGraph, AphelionError>                                          │
│                                                                             │
│  BuildGraph {                                                               │
│      nodes: [                                                               │
│          GraphNode { id: 1, name: "input", config: {...} },                 │
│          GraphNode { id: 2, name: "hidden", config: {...} },                │
│          GraphNode { id: 3, name: "output", config: {...} },                │
│      ],                                                                     │
│      edges: [(1, 2), (2, 3)],                                               │
│  }                                                                          │
│                                                                             │
│  stable_hash() = "a1b2c3d4e5f6..."                                          │
│                                                                             │
│  trace.events() = [                                                         │
│      TraceEvent { id: "pipeline.validate", ... },                           │
│      TraceEvent { id: "pipeline.start", ... },                              │
│      TraceEvent { id: "pipeline.finish", ... },                             │
│  ]                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Extension Points

Aphelion is designed for extensibility at multiple levels:

### 1. Custom Backends

Implement the `Backend` trait to add new execution environments:

```rust
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;
    fn device(&self) -> &str;
}

// Example: Custom TPU backend
struct TpuBackend {
    device_id: u32,
}

impl Backend for TpuBackend {
    fn name(&self) -> &str { "tpu" }
    fn device(&self) -> &str { "tpu:0" }
}
```

### 2. Custom TraceSinks

Implement `TraceSink` for custom logging/monitoring:

```rust
pub trait TraceSink: Send + Sync {
    fn record(&self, event: TraceEvent);
}

// Example: Send events to external monitoring service
struct PrometheusTraceSink { /* ... */ }

impl TraceSink for PrometheusTraceSink {
    fn record(&self, event: TraceEvent) {
        // Push metrics to Prometheus
    }
}
```

### 3. Custom Validators

Implement `ConfigValidator` for domain-specific validation:

```rust
pub trait ConfigValidator: Send + Sync {
    fn validate(&self, config: &ModelConfig) -> Result<(), Vec<ValidationError>>;
}

// Example: Validate layer configurations
struct LayerValidator;

impl ConfigValidator for LayerValidator {
    fn validate(&self, config: &ModelConfig) -> Result<(), Vec<ValidationError>> {
        // Check layer parameters are valid
    }
}
```

### 4. Custom Model Builders

Manually implement `ModelBuilder` for full control:

```rust
pub trait ModelBuilder: Send + Sync {
    type Output;
    fn build(&self, backend: &dyn Backend, trace: &dyn TraceSink) -> Self::Output;
}
```

### Extension Points Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXTENSION POINTS                                     │
├──────────────────┬────────────────────────┬─────────────────────────────────┤
│  Extension       │  Trait                 │  Purpose                        │
├──────────────────┼────────────────────────┼─────────────────────────────────┤
│  Backend         │  Backend               │  New compute targets            │
│  Tracing         │  TraceSink             │  Custom diagnostics             │
│  Validation      │  ConfigValidator       │  Domain-specific checks         │
│  Model Building  │  ModelBuilder          │  Custom build logic             │
│  Configuration   │  ConfigSpec            │  Config extraction              │
│  Export          │  SerializableTraceEvent│  Custom serialization           │
└──────────────────┴────────────────────────┴─────────────────────────────────┘
```

---

## Feature Flags Explained

Aphelion uses Cargo feature flags to enable optional functionality:

```toml
[features]
default = []

# Backend integrations
burn = []           # Burn deep learning framework
cubecl = []         # CubeCL GPU compute library
rust-ai-core = []   # Rust AI Core integration
tritter-accel = []  # Tritter acceleration

# Runtime features
tokio = ["dep:tokio"]  # Async runtime support
```

### Feature Details

| Feature | Status | Description |
|---------|--------|-------------|
| `burn` | Placeholder | Enables the Burn deep learning backend. Provides `BurnBackend` with support for CPU (NdArray), CUDA, Metal, and Vulkan devices. |
| `cubecl` | Placeholder | Enables the CubeCL GPU compute backend. Provides `CubeclBackend` with support for writing portable GPU kernels targeting CUDA, Metal, Vulkan, and WebGPU. |
| `rust-ai-core` | Placeholder | Future integration with the Rust AI Core ecosystem. |
| `tritter-accel` | Placeholder | Enables Tritter acceleration support for optimized model execution. |
| `tokio` | Available | Enables async runtime support using Tokio. Required for async pipeline operations. |

### Feature Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FEATURE FLAGS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  default ─────────────────────────────────────────────────────── (none)     │
│                                                                             │
│  burn ────────────────────────────────────────────────────────── burn_backend│
│    │                                                                        │
│    └─► Enables: BurnBackend, BurnDevice, BurnBackendConfig                  │
│                                                                             │
│  cubecl ──────────────────────────────────────────────────────── cubecl_backend│
│    │                                                                        │
│    └─► Enables: CubeclBackend, CubeclDevice, CubeclBackendConfig            │
│                                                                             │
│  rust-ai-core ────────────────────────────────────────────────── rust_ai_core│
│    │                                                                        │
│    └─► Enables: (future) integration types                                  │
│                                                                             │
│  tritter-accel ───────────────────────────────────────────────── tritter_backend│
│    │                                                                        │
│    └─► Enables: (future) Tritter acceleration                               │
│                                                                             │
│  tokio ───────────────────────────────────────────────────────── dep:tokio  │
│    │                                                                        │
│    └─► Enables: Async pipeline operations                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Usage Examples

**Basic usage (no features):**
```toml
[dependencies]
aphelion-core = "1.0"
```

**With Burn backend:**
```toml
[dependencies]
aphelion-core = { version = "1.0", features = ["burn"] }
```

**Multiple backends:**
```toml
[dependencies]
aphelion-core = { version = "1.0", features = ["burn", "cubecl"] }
```

**Full featured:**
```toml
[dependencies]
aphelion-core = { version = "1.0", features = ["burn", "cubecl", "tokio"] }
```

---

## Appendix: Type Reference

### Core Types

| Type | Module | Description |
|------|--------|-------------|
| `ModelConfig` | `config` | Model configuration container |
| `ConfigSpec` | `config` | Trait for types with configuration |
| `BuildGraph` | `graph` | Computation graph structure |
| `GraphNode` | `graph` | Single node in the graph |
| `NodeId` | `graph` | Unique node identifier |
| `Backend` | `backend` | Compute backend trait |
| `ModelBuilder` | `backend` | Model building trait |
| `NullBackend` | `backend` | Default/testing backend |
| `TraceEvent` | `diagnostics` | Single trace event |
| `TraceSink` | `diagnostics` | Trace collection trait |
| `InMemoryTraceSink` | `diagnostics` | In-memory trace storage |
| `BuildPipeline` | `pipeline` | Build orchestration |
| `BuildContext` | `pipeline` | Build context container |
| `AphelionError` | `error` | Error enumeration |
| `AphelionResult<T>` | `error` | Result type alias |

### Validation Types

| Type | Module | Description |
|------|--------|-------------|
| `ValidationError` | `validation` | Single validation failure |
| `ConfigValidator` | `validation` | Validator trait |
| `NameValidator` | `validation` | Model name validator |
| `VersionValidator` | `validation` | Semantic version validator |

### Export Types

| Type | Module | Description |
|------|--------|-------------|
| `SerializableTraceEvent` | `export` | JSON-serializable event |
| `JsonExporter` | `export` | JSON export trace sink |

### Backend-Specific Types (Feature-Gated)

| Type | Feature | Description |
|------|---------|-------------|
| `BurnBackend` | `burn` | Burn framework backend |
| `BurnDevice` | `burn` | Burn device selector |
| `BurnBackendConfig` | `burn` | Burn configuration |
| `CubeclBackend` | `cubecl` | CubeCL backend |
| `CubeclDevice` | `cubecl` | CubeCL device selector |
| `CubeclBackendConfig` | `cubecl` | CubeCL configuration |

---

*This document describes Aphelion Framework v1.0.0*
