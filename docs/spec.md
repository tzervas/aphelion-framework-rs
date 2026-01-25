# Aphelion Framework Spec (Expanded)

This document expands the canonical spec in [SPEC.md](../SPEC.md).

## 1) System Goals
- **Transparency**: every build step is traceable.
- **Repeatability**: identical configs produce identical graphs and hashes.
- **Performance**: backends are pluggable; burn is the primary backend.
- **Ease of Use**: macro-driven builder ergonomics and safe defaults.

## 2) Module Contracts

### 2.1 config
- `ModelConfig` must be serializable and deterministic (BTreeMap ordering).
- Validation requires non-empty name and version.

### 2.2 graph
- `BuildGraph` must have a deterministic `stable_hash()`.
- Node/edge addition must preserve insertion ordering.

### 2.3 diagnostics
- `TraceEvent` must have `id`, `message`, `timestamp`.
- `TraceSink` must be thread-safe (`Send + Sync`).

### 2.4 backend
- `Backend` must expose `name()` and `device()`.
- Default `NullBackend` is used for tests and examples.

### 2.5 pipeline
- `BuildPipeline::build_with_validation` must validate `ModelConfig`.
- Pipeline must emit start/validate/finish trace events.

### 2.6 macros
- `#[aphelion_model]` requires a `config` field.
- `build_graph(&self, backend, trace)` convention auto-implements `ModelBuilder`.

### 2.7 burn backend (first)
- `BurnBackendConfig` must be constructible with defaults.
- Device selection must be represented (CPU, CUDA, Metal, Vulkan).

## 3) Success Criteria Details

### SC-1: Deterministic graph hash
- **Test**: `graph_hash_is_deterministic`
- **Docs**: this section + `BuildGraph::stable_hash()` design.

### SC-2: Config validation enforced
- **Test**: `pipeline_validates_config`
- **Docs**: non-empty name/version rules; extendable later.

### SC-3: Trace capture works
- **Test**: `trace_sink_records_events` (to be implemented).
- **Docs**: trace schema and ordering guarantees.

### SC-4: Macro ergonomic contract
- **Test**: `macro_build_graph_convention` (to be implemented).
- **Docs**: `config` field + `build_graph` method required.

### SC-5: Burn-first backend availability
- **Test**: `burn_backend_config_defaults` (to be implemented).
- **Docs**: `BurnBackendConfig` must include device + perf toggles.

## 4) Future Extensions
- Rust-AI-core adapter mapping for configs and graphs.
- CubeCL backend integration.
- Trace exporters (JSON, OTEL).
