//! # aphelion-core
//!
//! Core library for the Aphelion AI Framework.
//!
//! ## Why This Crate Exists
//!
//! Aphelion is a framework frontend that provides an easier entrypoint for AI
//! engineering in Rust. This crate provides the core infrastructure that unifies
//! access to Rust AI libraries through a consistent API.
//!
//! Building AI systems means integrating multiple libraries: tensor operations,
//! memory management, device handling, training loops. aphelion-core handles:
//!
//! - **Unified API**: One consistent interface to underlying AI libraries
//!   (rust-ai-core, Candle, Burn, CubeCL).
//!
//! - **Reusable Components**: Configurations and pipelines can be templated,
//!   shared, and versioned for reproducible experiments.
//!
//! - **Deterministic graph hashing**: SHA-256 over canonicalized node data ensures
//!   identical configurations produce identical hashes, regardless of construction order.
//!
//! - **Structured tracing**: Every operation emits typed events. No printf debugging.
//!   Export to JSON for analysis or feed to observability systems.
//!
//! - **Backend abstraction**: Write once, run on CPU, GPU, or accelerators. The
//!   `Backend` trait abstracts hardware differences.
//!
//! - **Pipeline composition**: Stages execute in order with hooks for customization.
//!   Errors propagate with context. Progress is observable.
//!
//! ## Module Organization
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`config`] | Model configuration with typed parameters and validation |
//! | [`graph`] | DAG representation of model architecture |
//! | [`pipeline`] | Stage-based execution with hooks and progress tracking |
//! | [`backend`] | Hardware abstraction for CPU/GPU/accelerator targets |
//! | [`diagnostics`] | Structured tracing and event logging |
//! | [`validation`] | Composable validators for configuration checking |
//! | [`error`] | Typed errors with context and chaining |
//! | [`export`] | Serialization of trace events to JSON |
//!
//! ## Feature Flags
//!
//! | Feature | Effect |
//! |---------|--------|
//! | `burn` | Enables Burn deep learning backend integration |
//! | `cubecl` | Enables CubeCL GPU compute backend |
//! | `rust-ai-core` | Enables memory tracking, device detection, dtype utilities |
//! | `cuda` | Enables CUDA support (requires `rust-ai-core`) |
//! | `tokio` | Enables async pipeline execution |
//! | `tritter-accel` | Enables Tritter hardware acceleration |
//! | `python` | Enables Python bindings via PyO3 |
//! | `wasm` | Enables WebAssembly/TypeScript bindings via wasm-bindgen |
//!
//! ## Quick Start
//!
//! ```rust
//! use aphelion_core::prelude::*;
//!
//! // Build a model graph
//! let mut graph = BuildGraph::default();
//! let encoder = graph.add_node("encoder", ModelConfig::new("enc", "1.0.0"));
//! let decoder = graph.add_node("decoder", ModelConfig::new("dec", "1.0.0"));
//! graph.add_edge(encoder, decoder);
//!
//! // Execute pipeline
//! let backend = NullBackend::cpu();
//! let trace = InMemoryTraceSink::new();
//! let ctx = BuildContext::new(&backend, &trace);
//! let result = BuildPipeline::standard().execute(&ctx, graph)?;
//!
//! // Hash is deterministic
//! assert_eq!(result.stable_hash().len(), 64); // SHA-256 hex
//! # Ok::<(), aphelion_core::error::AphelionError>(())
//! ```

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

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use aphelion_macros::aphelion_model;

// ============================================================================
// rust-ai-core Re-exports (when feature enabled)
// ============================================================================

// Re-export AphelionDevice based on feature
#[cfg(feature = "rust-ai-core")]
pub use rust_ai_core::AphelionDevice;

#[cfg(not(feature = "rust-ai-core"))]
pub use rust_ai_core::placeholder::AphelionDevice;

// Re-export MemoryTracker based on feature
#[cfg(feature = "rust-ai-core")]
pub use rust_ai_core::real::MemoryTracker;

#[cfg(not(feature = "rust-ai-core"))]
pub use rust_ai_core::placeholder::MemoryTracker;

// Re-export real rust-ai-core types when enabled
#[cfg(feature = "rust-ai-core")]
pub use rust_ai_core::real::{
    bytes_per_element, estimate_attention_memory, estimate_tensor_bytes, get_device, init_logging,
    is_floating_point, warn_if_cpu, DType, DTypeExt, Dequantize, Device, DeviceConfig, LogConfig,
    PrecisionMode, Quantize, Tensor, ValidatableConfig, DEFAULT_OVERHEAD_FACTOR, RAC_VERSION,
};

// Re-export GPU types from trit-vsa (requires cuda feature)
#[cfg(all(feature = "rust-ai-core", feature = "cuda"))]
pub use trit_vsa::gpu::{GpuDispatchable, GpuError, GpuResult};

// Re-export CubeCL types when both features enabled
#[cfg(all(feature = "rust-ai-core", feature = "cuda"))]
pub use rust_ai_core::cubecl::{
    allocate_output_buffer, candle_to_cubecl_handle, cubecl_to_candle_tensor,
    has_cubecl_cuda_support, CubeclContext, TensorBuffer,
};
