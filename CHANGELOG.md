# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
