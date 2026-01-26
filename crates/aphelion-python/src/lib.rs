//! Python bindings for the Aphelion AI Framework.
//!
//! This crate provides Python bindings for aphelion-core via PyO3,
//! enabling Python developers to use the Aphelion AI framework with
//! the same performance and reliability as Rust.
//!
//! ## Features
//!
//! - `rust-ai-core`: Enable memory tracking, device detection, and dtype utilities
//! - `cuda`: Enable CUDA GPU support (requires `rust-ai-core`)
//! - `burn`: Enable Burn backend support
//! - `cubecl`: Enable CubeCL backend support

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

/// Aphelion Python module.
#[pymodule]
fn aphelion(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
