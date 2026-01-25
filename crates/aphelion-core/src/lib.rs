//! Aphelion core APIs: configuration, tracing, graph building, and backend abstractions.

pub mod backend;
pub mod config;
pub mod diagnostics;
pub mod error;
pub mod export;
pub mod graph;
pub mod pipeline;
pub mod prelude;
pub mod rust_ai_core;
pub mod validation;

#[cfg(feature = "burn")]
pub mod burn_backend;

#[cfg(feature = "cubecl")]
pub mod cubecl_backend;

#[cfg(feature = "tritter-accel")]
pub mod acceleration;

#[cfg(feature = "tritter-accel")]
pub mod tritter_backend;

pub use aphelion_macros::aphelion_model;
