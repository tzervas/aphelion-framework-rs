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
    is_floating_point, warn_if_cpu, DType, DTypeExt, Dequantize, Device, DeviceConfig,
    GpuDispatchable, LogConfig, PrecisionMode, Quantize, Tensor, ValidatableConfig,
    DEFAULT_OVERHEAD_FACTOR, RAC_VERSION,
};

// Re-export CubeCL types when both features enabled
#[cfg(all(feature = "rust-ai-core", feature = "cuda"))]
pub use rust_ai_core::cubecl::{
    allocate_output_buffer, candle_to_cubecl_handle, cubecl_to_candle_tensor,
    has_cubecl_cuda_support, CubeclContext, TensorBuffer,
};
