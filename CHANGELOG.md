# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
