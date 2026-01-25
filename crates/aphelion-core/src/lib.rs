//! Aphelion Core: A comprehensive framework for building, validating, and executing AI model pipelines.
//!
//! Aphelion Core provides a complete abstraction for model graph construction, backend management,
//! configuration handling, and diagnostic tracing. It enables developers to build complex AI
//! workflows with a composable, type-safe architecture.
//!
//! # Key Components
//!
//! - **Configuration** (`config`): Type-safe model configuration with semantic versioning support
//! - **Graphs** (`graph`): Directed acyclic graph (DAG) representation of model architectures
//! - **Pipelines** (`pipeline`): Extensible build pipeline with stages, hooks, and progress tracking
//! - **Validation** (`validation`): Comprehensive configuration validation framework
//! - **Backends** (`backend`): Abstract backend interface for hardware-specific implementations
//! - **Diagnostics** (`diagnostics`): Tracing and event recording for debugging and analysis
//! - **Export** (`export`): JSON serialization and export of trace events
//! - **Error Handling** (`error`): Unified error type and result type for all operations
//!
//! # Quick Start
//!
//! ```ignore
//! use aphelion_core::prelude::*;
//!
//! // 1. Create a model configuration
//! let config = ModelConfig::new("my-model", "1.0.0");
//!
//! // 2. Build a computation graph
//! let mut graph = BuildGraph::default();
//! let node = graph.add_node("input", config);
//!
//! // 3. Set up backend and diagnostics
//! let backend = NullBackend::cpu();
//! let trace_sink = InMemoryTraceSink::new();
//!
//! // 4. Execute a pipeline
//! let pipeline = BuildPipeline::new()
//!     .with_stage(Box::new(ValidationStage))
//!     .with_stage(Box::new(HashingStage));
//!
//! let ctx = BuildContext {
//!     backend: &backend,
//!     trace: &trace_sink,
//! };
//!
//! let result = pipeline.execute(&ctx, graph)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Error Handling
//!
//! All operations return `AphelionResult<T>`, which is an alias for `Result<T, AphelionError>`.
//! Errors are categorized by type for comprehensive error handling:
//!
//! - `InvalidConfig`: Configuration validation failures
//! - `Backend`: Backend initialization or execution errors
//! - `Build`: Pipeline or model building errors
//! - `Validation`: Data validation errors
//! - `Serialization`: JSON serialization errors
//! - `Io`: I/O operation errors
//! - `Graph`: Graph-related errors (cycles, invalid structure)

pub mod backend;
pub mod config;
pub mod diagnostics;
pub mod error;
pub mod export;
pub mod graph;
pub mod pipeline;
pub mod prelude;
pub mod validation;

#[cfg(feature = "rust-ai-core")]
pub mod rust_ai_core;

#[cfg(feature = "burn")]
pub mod burn_backend;

#[cfg(feature = "cubecl")]
pub mod cubecl_backend;

pub use aphelion_macros::aphelion_model;
