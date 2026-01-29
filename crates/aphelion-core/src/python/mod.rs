//! Python bindings for the Aphelion AI Framework.
//!
//! This module provides Python bindings for aphelion-core via PyO3,
//! enabling Python developers to use Aphelion as a frontend for AI
//! engineering tasks.
//!
//! ## Why Python Bindings
//!
//! Aphelion is a framework frontend that provides an easier entrypoint for AI
//! development. These bindings bring that same unified API to Python:
//!
//! - **Unified API**: Consistent interface to Rust AI libraries from Python
//! - **Reusable Components**: Template and share configurations across projects
//! - **Deterministic Builds**: Same config = same hash = reproducible results
//! - **Integration**: Works alongside existing Python ML workflows
//!
//! ## Module Structure
//!
//! All types are exported at the top level:
//!
//! - `ModelConfig`, `ModelConfigBuilder`: Configuration management
//! - `BuildGraph`, `GraphNode`, `NodeId`: DAG representation
//! - `BuildPipeline`, `BuildContext`: Pipeline execution
//! - `NullBackend`: Reference backend for testing
//! - `InMemoryTraceSink`, `TraceEvent`, `TraceLevel`: Diagnostics
//! - `ValidationError`, `NameValidator`, `VersionValidator`, `CompositeValidator`: Validation
//!
//! ## Feature Flags
//!
//! | Feature | Effect |
//! |---------|--------|
//! | `rust-ai-core` | Memory tracking, device detection, dtype utilities |
//! | `cuda` | CUDA GPU support (requires `rust-ai-core`) |
//! | `burn` | Burn backend support |
//! | `cubecl` | CubeCL backend support |
//!
//! ## Example
//!
//! ```python
//! import aphelion
//!
//! # Build configuration
//! config = aphelion.ModelConfig("transformer", "1.0.0")
//! config = config.with_param("d_model", 512)
//!
//! # Build graph
//! graph = aphelion.BuildGraph()
//! encoder = graph.add_node("encoder", config)
//!
//! # Execute pipeline
//! ctx = aphelion.BuildContext.with_null_backend()
//! pipeline = aphelion.BuildPipeline.standard()
//! result = pipeline.execute(ctx, graph)
//!
//! print(f"Hash: {result.stable_hash()}")
//! ```

// Allow clippy false positive with PyO3's PyResult in return types
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

mod backend;
mod config;
mod diagnostics;
mod graph;
mod pipeline;
mod validation;

#[cfg(feature = "rust-ai-core")]
mod core;

/// Aphelion Python module - transparent, traceable AI model construction.
///
/// This module provides Python bindings for the Aphelion AI Framework,
/// enabling deterministic model graph construction with cryptographic
/// hashing for reproducibility.
///
/// Quick start:
///     >>> import aphelion
///     >>> config = aphelion.ModelConfig("encoder", "1.0.0")
///     >>> graph = aphelion.BuildGraph()
///     >>> node = graph.add_node("encoder", config)
///     >>> print(graph.stable_hash())
///
/// Module attributes:
///     __version__: Package version string
///     HAS_BURN: True if Burn backend is available
///     HAS_CUBECL: True if CubeCL backend is available
///     HAS_RUST_AI_CORE: True if rust-ai-core integration is available
///     HAS_CUDA: True if CUDA support is available
#[pymodule]
pub fn aphelion(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register all submodule types
    config::register(m)?;
    graph::register(m)?;
    backend::register(m)?;
    diagnostics::register(m)?;
    pipeline::register(m)?;
    validation::register(m)?;

    // Register rust-ai-core integration when feature enabled
    #[cfg(feature = "rust-ai-core")]
    core::register(m)?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Feature flags - expose what backends are compiled in
    #[cfg(feature = "burn")]
    m.add("HAS_BURN", true)?;
    #[cfg(not(feature = "burn"))]
    m.add("HAS_BURN", false)?;

    #[cfg(feature = "cubecl")]
    m.add("HAS_CUBECL", true)?;
    #[cfg(not(feature = "cubecl"))]
    m.add("HAS_CUBECL", false)?;

    #[cfg(feature = "rust-ai-core")]
    m.add("HAS_RUST_AI_CORE", true)?;
    #[cfg(not(feature = "rust-ai-core"))]
    m.add("HAS_RUST_AI_CORE", false)?;

    #[cfg(feature = "cuda")]
    m.add("HAS_CUDA", true)?;
    #[cfg(not(feature = "cuda"))]
    m.add("HAS_CUDA", false)?;

    #[cfg(feature = "tritter-accel")]
    m.add("HAS_TRITTER_ACCEL", true)?;
    #[cfg(not(feature = "tritter-accel"))]
    m.add("HAS_TRITTER_ACCEL", false)?;

    Ok(())
}
