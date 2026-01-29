# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.9] - 2026-01-28

### Added
- **TypeScript/WASM bindings**: New `wasm` feature providing WebAssembly bindings via wasm-bindgen
- npm package `aphelion-framework` for browser and Node.js usage
- WASM bindings for all core types:
  - `ModelConfig` with JSON serialization and presets
  - `BuildGraph`, `GraphNode`, `NodeId` with stable hashing
  - `BuildPipeline`, `BuildContext` for pipeline execution
  - `NullBackend`, `DeviceCapabilities` for backend abstraction
  - `TraceEvent`, `InMemoryTraceSink`, `TraceLevel` for diagnostics
  - `ValidationError`, `NameValidator`, `VersionValidator`, `CompositeValidator`
- Feature detection: `getVersion()`, `hasBurn()`, `hasCubecl()`, `hasRustAiCore()`
- Automatic panic hook for better error messages in development

### Changed
- Updated CI/release workflow with WASM build and npm publish jobs
- Updated README with TypeScript/JavaScript usage examples
- Updated feature flags documentation

### Usage
```typescript
import init, { ModelConfig, BuildGraph } from 'aphelion-framework';

await init();
const config = new ModelConfig("model", "1.0.0");
const graph = new BuildGraph();
graph.addNode("encoder", config);
console.log(graph.stableHash());
```

## [1.2.8] - 2026-01-29

### Changed
- **Unified Python bindings**: Moved from separate `aphelion-python` crate into `aphelion-core` with `python` feature
- Updated rust-ai-core: 0.2.7 → 0.3.1 (pyo3 0.27.2 compatibility via tritter-accel 0.1.3)
- Updated candle-core: 0.9 → 0.9.2
- Simplified CI/release pipeline (single crate for Rust lib + Python wheels)

### Removed
- `aphelion-python` crate (functionality now in `aphelion-core --features python`)

### Migration
Python package installation is unchanged:
```bash
pip install aphelion-framework
```

For wheel building from source:
```bash
cd crates/aphelion-core
maturin build --features python
```

## [1.2.7] - 2026-01-29

### Changed
- Updated rust-ai-core: 0.2.7 → 0.3.1
- Updated candle-core: 0.9 → 0.9.2
- tritter-accel updated to 0.1.3 with pyo3 0.27.2 compatibility

## [1.2.6] - 2026-01-26

### Fixed
- Python package now uses dynamic versioning from Cargo.toml (was stuck at 1.2.2)
- PyPI releases will now match crates.io versions automatically

## [1.2.5] - 2026-01-26

### Changed
- Updated dependencies to latest patch versions:
  - burn: 0.20 → 0.20.1
  - pyo3: 0.27 → 0.27.2
  - thiserror: 2.0 → 2.0.18
  - half: 2.7 → 2.7.1
- Updated qlora-candle fork to latest commit

## [1.2.4] - 2026-01-26

### Security - Unmaintained Dependency Fixes

This release replaces unmaintained transitive dependencies with actively maintained forks:

- **paste** (unmaintained) → **qlora-paste v1.0.20** (maintained fork)
- **gemm** (unmaintained) → **qlora-gemm v0.20.0** (maintained fork)
- **candle-core** patched via **qlora-candle** to use maintained dependencies

### Changed
- Added `[patch.crates-io]` for candle-core pointing to qlora-candle fork
- Dependency chain now uses maintained forks:
  ```
  candle-core (qlora-candle) -> qlora-gemm v0.20.0 -> qlora-paste v1.0.20
  ```
- Added qlora-paste v1.0.20 as direct dependency in aphelion-macros
- Added qlora-gemm v0.20.0 to workspace dependencies
- Updated security workflow to handle advisory warnings appropriately
- Added deny.toml ignore entries for tracked transitive advisories

### Upstream Contributions
- PR to huggingface/candle for upstream adoption: https://github.com/huggingface/candle/pull/3335
- Once merged, the `[patch.crates-io]` can be removed

### Note
For other projects wanting to adopt these maintained forks, add to Cargo.toml:
```toml
[patch.crates-io]
candle-core = { git = "https://github.com/tzervas/qlora-candle.git", branch = "use-qlora-gemm" }
```

## [1.2.3] - 2026-01-26

### Changed
- Removed broken git patch for paste crate (patch-target branch no longer exists)
- Added documentation comment noting paste is a transitive dependency from burn/cubecl
- When upstream burn/cubecl adopt qlora-paste, dependencies can be updated accordingly

### Note
- qlora-paste v1.0.17+ is now available on crates.io as a maintained fork
- Direct replacement requires import changes (`use qlora_paste::paste` vs `use paste::paste`)
- Waiting for upstream adoption in burn/cubecl ecosystem

## [1.2.2] - 2026-01-26

### Changed
- Updated all dependencies to latest stable versions:
  - rust-ai-core: 0.2.6 -> 0.2.7
  - pyo3: 0.22 -> 0.27 (Python bindings)
  - pythonize: 0.22 -> 0.27
  - pyo3-async-runtimes: 0.22 -> 0.27
  - thiserror: 1.x -> 2.0
  - burn: 0.16 -> 0.20 (now enabled as optional dependency)
  - cubecl: 0.4 -> 0.9 (now enabled as optional dependency)
- Python support updated to 3.10-3.14 (dropped 3.9)
- MSRV set to Rust 1.92
- Fixed deprecated PyObject usage (now Py<PyAny>)
- Fixed pyo3 0.27 API changes in core.rs (PyDict instead of HashMap)

### Security
- cargo audit: 3 transitive dependency warnings (bincode, paste, lru - all from burn/cubecl)
- No direct vulnerabilities in aphelion crates

## [1.2.1] - 2026-01-25

### Added
- Comprehensive Python docstrings for all binding classes (IDE support, help())
- Module-level documentation explaining "why" for all core modules

### Changed
- Repositioned framework as a "frontend" providing unified API to Rust AI ecosystem
- Documentation now emphasizes reusable, templatable components
- README rewritten to focus on framework frontend role

### Fixed
- Python pipeline execute() now runs actual stages instead of passthrough
- BuildContext struct literal usage updated to use constructors

## [1.1.0] - 2026-01-25

### Added
- rust-ai-core v0.2.6 integration with memory tracking and device detection
- Python bindings via PyO3 with async support (pyo3-async-runtimes)
- Typed parameter retrieval: `config.param::<T>()`, `param_or()`, `param_or_default()`
- Preset pipelines: `BuildPipeline::standard()`, `for_training()`, `for_inference()`
- TraceSinkExt helper methods: `info()`, `warn()`, `error()`, `debug()`
- BuildContext constructors: `new()`, `with_null_backend()`
- Rich error context: `in_stage()`, `for_field()` methods
- Backend auto-detection via rust-ai-core device utilities

### Changed
- Updated workspace dependencies for rust-ai-core ecosystem
- Release workflow uses `cargo test --workspace` (not --all-features)

### Fixed
- pyo3 version conflict resolved (rust-ai-core-bindings separate package)
- CI workflow CUDA build issue fixed

## [1.0.0] - 2026-01-25

### Added

#### Core Framework
- `ModelConfig` with semantic versioning and deterministic parameter storage (BTreeMap)
- `ModelConfigBuilder` with type-safe parameter setters and presets (small/medium/large)
- `BuildGraph` with stable SHA256 hashing for reproducible builds
- Graph traversal utilities (BFS, topological sort) and DOT export
- Comprehensive error types via `AphelionError` enum

#### Backend Abstraction
- `Backend` trait with device capabilities and availability checks
- `NullBackend` for testing and examples
- `MockBackend` for configurable test scenarios
- `BackendRegistry` for runtime backend management
- `DeviceCapabilities` and `MemoryInfo` structs

#### Pipeline System
- `PipelineStage` trait for extensible build stages
- `BuildPipeline` with stage composition, hooks, and skip functionality
- `ValidationStage` and `HashingStage` built-in implementations
- Progress callback support via `ProgressCallback`

#### Diagnostics
- `TraceEvent` with levels, timestamps, and distributed tracing IDs
- `TraceSink` trait with `InMemoryTraceSink` implementation
- `TraceFilter` for level-based filtering
- `MultiSink` for simultaneous multi-output tracing
- JSON export via `ExportedTrace`

#### Validation
- `ConfigValidator` trait with composable validators
- `RequiredFieldValidator` and `VersionValidator` implementations
- `CompositeValidator` for combining multiple validators

#### Macros
- `#[aphelion_model]` attribute macro with automatic trait implementations
- Support for `#[skip]` and `#[rename]` field attributes
- Auto-implements `ModelBuilder` for structs with `config` field and `build_graph` method

#### Examples
- `basic_usage`: Simple model building workflow
- `custom_backend`: Implementing custom backend trait
- `pipeline_stages`: Creating and composing pipeline stages
- `validation`: Configuration validation patterns

### Success Criteria Met
- SC-1: Deterministic graph hash (test: `graph_hash_is_deterministic`)
- SC-2: Config validation enforced (test: `pipeline_validates_config`)
- SC-3: Trace capture works (test: `trace_sink_records_events`)
- SC-4: Macro ergonomic contract (test: `macro_build_graph_convention`)
- SC-5: Burn-first backend availability (test: `burn_backend_config_defaults`)

### Feature Flags
- `burn` - Burn backend integration (placeholder)
- `cubecl` - CubeCL integration (placeholder)
- `rust-ai-core` - rust-ai-core integration (placeholder)
