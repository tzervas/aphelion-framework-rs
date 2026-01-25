//! # rust-ai-core Integration Module
//!
//! This module provides integration with the `rust-ai-core` crate for device
//! management, memory tracking, dtype utilities, and CubeCL interoperability.
//!
//! ## Feature Gates
//!
//! - `rust-ai-core`: Enables real rust-ai-core integration with actual types
//! - `cuda`: Enables CubeCL CUDA support (requires `rust-ai-core`)
//!
//! When the `rust-ai-core` feature is disabled, placeholder types are provided
//! for API compatibility during development.

use crate::config::ModelConfig;
use crate::error::{AphelionError, AphelionResult};
use crate::graph::{BuildGraph, NodeId};
use std::collections::BTreeMap;

// ============================================================================
// Aphelion Adapter Types (always available)
// ============================================================================

/// Device abstraction for rust-ai-core integration.
///
/// This type represents a compute device that can be used for model execution.
/// When `rust-ai-core` is enabled, this maps to actual hardware. Otherwise,
/// it provides a placeholder for API compatibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RacDevice {
    /// Device identifier (e.g., "cuda:0", "cpu")
    pub id: String,
    /// Device type classification
    pub device_type: RacDeviceType,
    /// Memory capacity in bytes (if applicable)
    pub memory_bytes: Option<u64>,
}

/// Device type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RacDeviceType {
    Cpu,
    Cuda,
    Rocm,
    OneApi,
    Metal,
    Vulkan,
    Custom,
}

impl RacDevice {
    /// Create a CPU device.
    pub fn default_cpu() -> Self {
        Self {
            id: "cpu:0".to_string(),
            device_type: RacDeviceType::Cpu,
            memory_bytes: None,
        }
    }

    /// Create a CUDA device.
    pub fn cuda(index: u32) -> Self {
        Self {
            id: format!("cuda:{}", index),
            device_type: RacDeviceType::Cuda,
            memory_bytes: None,
        }
    }

    /// Set memory capacity.
    pub fn with_memory(mut self, bytes: u64) -> Self {
        self.memory_bytes = Some(bytes);
        self
    }
}

/// Model configuration in rust-ai-core format.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RacModelConfig {
    pub name: String,
    pub version: String,
    pub batch_size: Option<u32>,
    pub sequence_length: Option<u32>,
    pub hidden_size: Option<u32>,
    pub num_attention_heads: Option<u32>,
    pub num_layers: Option<u32>,
    pub vocab_size: Option<u32>,
    pub dtype: RacDataType,
    pub custom_params: BTreeMap<String, String>,
}

/// Data type for computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RacDataType {
    #[default]
    Float32,
    Float16,
    BFloat16,
    Float64,
    Int32,
    Int64,
    Int8,
    UInt8,
}

/// Node handle in a rust-ai-core compute graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RacNodeHandle {
    pub index: u64,
    pub generation: u32,
}

impl From<NodeId> for RacNodeHandle {
    fn from(id: NodeId) -> Self {
        Self {
            index: id.value(),
            generation: 0,
        }
    }
}

/// Compute graph in rust-ai-core format.
#[derive(Debug, Clone, Default)]
pub struct RacComputeGraph {
    pub nodes: Vec<RacGraphNode>,
    pub edges: Vec<(RacNodeHandle, RacNodeHandle)>,
    pub metadata: RacGraphMetadata,
}

/// A node in the compute graph.
#[derive(Debug, Clone)]
pub struct RacGraphNode {
    pub handle: RacNodeHandle,
    pub op_type: String,
    pub config: RacModelConfig,
    pub input_shapes: Vec<Vec<i64>>,
    pub output_shapes: Vec<Vec<i64>>,
}

/// Metadata for a compute graph.
#[derive(Debug, Clone, Default)]
pub struct RacGraphMetadata {
    pub source_framework: String,
    pub content_hash: String,
    pub is_optimized: bool,
    pub device_hints: Vec<String>,
}

// ============================================================================
// Real rust-ai-core Integration (when feature enabled)
// ============================================================================

#[cfg(feature = "rust-ai-core")]
pub mod real {
    //! Real rust-ai-core types and re-exports.

    // Re-export Device management
    pub use rust_ai_core::{get_device, warn_if_cpu, DeviceConfig};

    // Re-export Memory tracking
    pub use rust_ai_core::memory::{
        estimate_attention_memory, estimate_tensor_bytes, MemoryTracker, DEFAULT_OVERHEAD_FACTOR,
    };

    // Re-export DType utilities
    pub use rust_ai_core::dtype::{bytes_per_element, is_floating_point, DTypeExt, PrecisionMode};

    // Re-export Error types
    pub use rust_ai_core::{CoreError, Result as RacResult};

    // Re-export Traits
    pub use rust_ai_core::{Dequantize, GpuDispatchable, Quantize, ValidatableConfig};

    // Re-export Logging
    pub use rust_ai_core::{init_logging, LogConfig};

    // Re-export Version
    pub use rust_ai_core::VERSION as RAC_VERSION;

    // Re-export candle types
    pub use candle_core::{DType, Device, Tensor};

    // CubeCL interop (requires cuda feature)
    #[cfg(feature = "cuda")]
    pub use rust_ai_core::{
        allocate_output_buffer, candle_to_cubecl_handle, cubecl_to_candle_tensor,
        has_cubecl_cuda_support, TensorBuffer,
    };
}

#[cfg(feature = "rust-ai-core")]
pub use real::*;

// ============================================================================
// Device Bridge (when rust-ai-core feature enabled)
// ============================================================================

#[cfg(feature = "rust-ai-core")]
mod device_bridge {
    use super::*;
    use candle_core::Device;
    use rust_ai_core::{get_device, warn_if_cpu, DeviceConfig};

    /// Bridge between aphelion's device abstraction and candle's Device.
    #[derive(Debug, Clone)]
    pub struct AphelionDevice {
        inner: Device,
        config: DeviceConfig,
    }

    impl AphelionDevice {
        /// Create a device from configuration.
        pub fn from_config(config: DeviceConfig) -> AphelionResult<Self> {
            let device = get_device(&config)
                .map_err(|e| AphelionError::backend(format!("Device selection failed: {}", e)))?;
            Ok(Self {
                inner: device,
                config,
            })
        }

        /// Create a CPU device.
        pub fn cpu() -> Self {
            Self {
                inner: Device::Cpu,
                config: DeviceConfig::new().with_force_cpu(true),
            }
        }

        /// Create a CUDA device.
        pub fn cuda(ordinal: usize) -> AphelionResult<Self> {
            let config = DeviceConfig::new().with_cuda_device(ordinal);
            Self::from_config(config)
        }

        /// Auto-select best available device.
        pub fn auto() -> AphelionResult<Self> {
            Self::from_config(DeviceConfig::default())
        }

        /// Get the underlying candle Device.
        pub fn as_candle_device(&self) -> &Device {
            &self.inner
        }

        /// Consume and return candle Device.
        pub fn into_candle_device(self) -> Device {
            self.inner
        }

        /// Check if CUDA device.
        pub fn is_cuda(&self) -> bool {
            matches!(self.inner, Device::Cuda(_))
        }

        /// Check if CPU device.
        pub fn is_cpu(&self) -> bool {
            matches!(self.inner, Device::Cpu)
        }

        /// Warn if running on CPU.
        pub fn warn_if_cpu(&self, crate_name: &str) {
            warn_if_cpu(&self.inner, crate_name);
        }

        /// Get device configuration.
        pub fn config(&self) -> &DeviceConfig {
            &self.config
        }
    }

    impl From<Device> for AphelionDevice {
        fn from(device: Device) -> Self {
            let config = match &device {
                Device::Cpu => DeviceConfig::new().with_force_cpu(true),
                Device::Cuda(_) => DeviceConfig::default(),
                _ => DeviceConfig::default(),
            };
            Self {
                inner: device,
                config,
            }
        }
    }
}

#[cfg(feature = "rust-ai-core")]
pub use device_bridge::AphelionDevice;

// ============================================================================
// CubeCL Context (when rust-ai-core + cuda features enabled)
// ============================================================================

#[cfg(all(feature = "rust-ai-core", feature = "cuda"))]
pub mod cubecl {
    //! CubeCL tensor interoperability utilities.

    use super::*;
    use candle_core::{DType, Device, Tensor};

    pub use rust_ai_core::{
        allocate_output_buffer, candle_to_cubecl_handle, cubecl_to_candle_tensor,
        has_cubecl_cuda_support, TensorBuffer,
    };

    /// Wrapper for CubeCL operations.
    pub struct CubeclContext {
        device: Device,
    }

    impl CubeclContext {
        /// Create a CubeCL context for a CUDA device.
        pub fn new(device: Device) -> AphelionResult<Self> {
            if !matches!(device, Device::Cuda(_)) {
                return Err(AphelionError::backend("CubeCL requires CUDA device"));
            }
            if !has_cubecl_cuda_support() {
                return Err(AphelionError::backend("CubeCL CUDA support not available"));
            }
            Ok(Self { device })
        }

        /// Get device reference.
        pub fn device(&self) -> &Device {
            &self.device
        }

        /// Convert tensor to CubeCL buffer.
        pub fn tensor_to_buffer(&self, tensor: &Tensor) -> AphelionResult<TensorBuffer> {
            candle_to_cubecl_handle(tensor)
                .map_err(|e| AphelionError::backend(format!("CubeCL conversion failed: {}", e)))
        }

        /// Convert CubeCL buffer to tensor.
        pub fn buffer_to_tensor(&self, buffer: &TensorBuffer) -> AphelionResult<Tensor> {
            cubecl_to_candle_tensor(buffer, &self.device)
                .map_err(|e| AphelionError::backend(format!("CubeCL conversion failed: {}", e)))
        }

        /// Allocate output buffer.
        pub fn alloc_output(&self, shape: &[usize], dtype: DType) -> AphelionResult<TensorBuffer> {
            allocate_output_buffer(shape, dtype)
                .map_err(|e| AphelionError::backend(format!("CubeCL allocation failed: {}", e)))
        }
    }
}

#[cfg(all(feature = "rust-ai-core", feature = "cuda"))]
pub use cubecl::CubeclContext;

// ============================================================================
// Placeholder Types (when rust-ai-core feature disabled)
// ============================================================================

#[cfg(not(feature = "rust-ai-core"))]
pub mod placeholder {
    //! Placeholder types when rust-ai-core is disabled.

    use super::*;

    /// Placeholder memory tracker.
    #[derive(Debug, Default)]
    pub struct MemoryTracker {
        allocated: std::sync::atomic::AtomicUsize,
        peak: std::sync::atomic::AtomicUsize,
        limit: usize,
        overhead_factor: f64,
    }

    impl Clone for MemoryTracker {
        fn clone(&self) -> Self {
            use std::sync::atomic::Ordering;
            Self {
                allocated: std::sync::atomic::AtomicUsize::new(
                    self.allocated.load(Ordering::SeqCst),
                ),
                peak: std::sync::atomic::AtomicUsize::new(self.peak.load(Ordering::SeqCst)),
                limit: self.limit,
                overhead_factor: self.overhead_factor,
            }
        }
    }

    impl MemoryTracker {
        pub fn new() -> Self {
            Self {
                allocated: std::sync::atomic::AtomicUsize::new(0),
                peak: std::sync::atomic::AtomicUsize::new(0),
                limit: usize::MAX,
                overhead_factor: 1.1,
            }
        }

        pub fn with_limit(limit: usize) -> Self {
            Self {
                allocated: std::sync::atomic::AtomicUsize::new(0),
                peak: std::sync::atomic::AtomicUsize::new(0),
                limit,
                overhead_factor: 1.1,
            }
        }

        pub fn with_overhead_factor(mut self, factor: f64) -> Self {
            self.overhead_factor = factor;
            self
        }

        pub fn allocate(&self, bytes: usize) -> AphelionResult<()> {
            use std::sync::atomic::Ordering;
            let current = self.allocated.fetch_add(bytes, Ordering::SeqCst) + bytes;
            if current > self.limit {
                self.allocated.fetch_sub(bytes, Ordering::SeqCst);
                return Err(AphelionError::backend(format!(
                    "Memory limit exceeded: {} > {}",
                    current, self.limit
                )));
            }
            self.peak.fetch_max(current, Ordering::SeqCst);
            Ok(())
        }

        pub fn deallocate(&self, bytes: usize) {
            use std::sync::atomic::Ordering;
            self.allocated.fetch_sub(bytes, Ordering::SeqCst);
        }

        pub fn would_fit(&self, bytes: usize) -> bool {
            use std::sync::atomic::Ordering;
            self.allocated.load(Ordering::SeqCst) + bytes <= self.limit
        }

        pub fn allocated_bytes(&self) -> usize {
            use std::sync::atomic::Ordering;
            self.allocated.load(Ordering::SeqCst)
        }

        pub fn peak_bytes(&self) -> usize {
            use std::sync::atomic::Ordering;
            self.peak.load(Ordering::SeqCst)
        }

        pub fn limit_bytes(&self) -> usize {
            self.limit
        }
    }

    /// Placeholder tensor bytes estimation.
    pub fn estimate_tensor_bytes(shape: &[usize], dtype: RacDataType) -> usize {
        let element_size = match dtype {
            RacDataType::Float32 | RacDataType::Int32 => 4,
            RacDataType::Float64 | RacDataType::Int64 => 8,
            RacDataType::Float16 | RacDataType::BFloat16 => 2,
            RacDataType::Int8 | RacDataType::UInt8 => 1,
        };
        shape.iter().product::<usize>() * element_size
    }

    pub const DEFAULT_OVERHEAD_FACTOR: f64 = 1.1;

    /// Placeholder device configuration.
    #[derive(Debug, Clone, Default)]
    pub struct DeviceConfig {
        force_cpu: bool,
        cuda_device: Option<usize>,
        crate_name: Option<String>,
    }

    impl DeviceConfig {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn with_force_cpu(mut self, force: bool) -> Self {
            self.force_cpu = force;
            self
        }

        pub fn with_cuda_device(mut self, ordinal: usize) -> Self {
            self.cuda_device = Some(ordinal);
            self
        }

        pub fn with_crate_name(mut self, name: impl Into<String>) -> Self {
            self.crate_name = Some(name.into());
            self
        }
    }

    /// Placeholder AphelionDevice.
    #[derive(Debug, Clone)]
    pub struct AphelionDevice {
        device: RacDevice,
        config: DeviceConfig,
    }

    impl AphelionDevice {
        pub fn from_config(config: DeviceConfig) -> AphelionResult<Self> {
            let device = if config.force_cpu {
                RacDevice::default_cpu()
            } else if let Some(ordinal) = config.cuda_device {
                RacDevice::cuda(ordinal as u32)
            } else {
                RacDevice::default_cpu()
            };
            Ok(Self { device, config })
        }

        pub fn cpu() -> Self {
            Self {
                device: RacDevice::default_cpu(),
                config: DeviceConfig::new().with_force_cpu(true),
            }
        }

        pub fn cuda(_ordinal: usize) -> AphelionResult<Self> {
            Err(AphelionError::backend(
                "CUDA not available (rust-ai-core feature disabled)",
            ))
        }

        pub fn auto() -> AphelionResult<Self> {
            Ok(Self::cpu())
        }

        pub fn is_cuda(&self) -> bool {
            self.device.device_type == RacDeviceType::Cuda
        }

        pub fn is_cpu(&self) -> bool {
            self.device.device_type == RacDeviceType::Cpu
        }

        pub fn warn_if_cpu(&self, crate_name: &str) {
            if self.is_cpu() {
                tracing::warn!(
                    "{}: Running on CPU. Consider enabling CUDA for better performance.",
                    crate_name
                );
            }
        }

        pub fn config(&self) -> &DeviceConfig {
            &self.config
        }
    }
}

#[cfg(not(feature = "rust-ai-core"))]
pub use placeholder::*;

// ============================================================================
// Adapter Traits
// ============================================================================

/// Trait for converting aphelion configurations to rust-ai-core format.
pub trait ConfigAdapter: Send + Sync {
    fn convert(&self, config: &ModelConfig) -> AphelionResult<RacModelConfig>;

    fn validate(&self, config: &ModelConfig) -> AphelionResult<()> {
        self.convert(config).map(|_| ())
    }

    fn supported_params(&self) -> &[&str] {
        &[
            "batch_size",
            "sequence_length",
            "hidden_size",
            "num_attention_heads",
            "num_layers",
            "vocab_size",
            "dtype",
        ]
    }
}

/// Trait for converting aphelion graphs to rust-ai-core format.
pub trait GraphAdapter: Send + Sync {
    fn convert(&self, graph: &BuildGraph) -> AphelionResult<RacComputeGraph>;

    fn convert_for_device(
        &self,
        graph: &BuildGraph,
        device: &RacDevice,
    ) -> AphelionResult<RacComputeGraph> {
        let mut rac_graph = self.convert(graph)?;
        rac_graph.metadata.device_hints.push(device.id.clone());
        Ok(rac_graph)
    }

    fn validate(&self, graph: &BuildGraph) -> AphelionResult<()>;
}

/// Trait for runtime execution.
pub trait RuntimeAdapter: Send + Sync {
    type Output;

    fn execute(&self, graph: &RacComputeGraph, device: &RacDevice)
        -> AphelionResult<Self::Output>;

    fn is_device_available(&self, device: &RacDevice) -> bool;

    fn available_devices(&self) -> Vec<RacDevice>;
}

// ============================================================================
// Default Implementations
// ============================================================================

/// Default configuration adapter.
#[derive(Debug, Clone, Default)]
pub struct DefaultConfigAdapter;

impl ConfigAdapter for DefaultConfigAdapter {
    fn convert(&self, config: &ModelConfig) -> AphelionResult<RacModelConfig> {
        let mut rac_config = RacModelConfig {
            name: config.name.clone(),
            version: config.version.clone(),
            ..Default::default()
        };

        if let Some(val) = config.params.get("batch_size") {
            rac_config.batch_size = val.as_u64().map(|v| v as u32);
        }
        if let Some(val) = config.params.get("sequence_length") {
            rac_config.sequence_length = val.as_u64().map(|v| v as u32);
        }
        if let Some(val) = config.params.get("hidden_size") {
            rac_config.hidden_size = val.as_u64().map(|v| v as u32);
        }
        if let Some(val) = config.params.get("num_attention_heads") {
            rac_config.num_attention_heads = val.as_u64().map(|v| v as u32);
        }
        if let Some(val) = config.params.get("num_layers") {
            rac_config.num_layers = val.as_u64().map(|v| v as u32);
        }
        if let Some(val) = config.params.get("vocab_size") {
            rac_config.vocab_size = val.as_u64().map(|v| v as u32);
        }
        if let Some(val) = config.params.get("dtype") {
            if let Some(dtype_str) = val.as_str() {
                rac_config.dtype = parse_dtype(dtype_str)?;
            }
        }

        for (key, val) in &config.params {
            if !self.supported_params().contains(&key.as_str()) {
                rac_config
                    .custom_params
                    .insert(key.clone(), val.to_string());
            }
        }

        Ok(rac_config)
    }
}

fn parse_dtype(s: &str) -> AphelionResult<RacDataType> {
    match s.to_lowercase().as_str() {
        "float32" | "f32" => Ok(RacDataType::Float32),
        "float16" | "f16" => Ok(RacDataType::Float16),
        "bfloat16" | "bf16" => Ok(RacDataType::BFloat16),
        "float64" | "f64" => Ok(RacDataType::Float64),
        "int32" | "i32" => Ok(RacDataType::Int32),
        "int64" | "i64" => Ok(RacDataType::Int64),
        "int8" | "i8" => Ok(RacDataType::Int8),
        "uint8" | "u8" => Ok(RacDataType::UInt8),
        _ => Err(AphelionError::config_error(format!("Unknown dtype: {}", s))),
    }
}

/// Default graph adapter.
#[derive(Debug, Clone)]
pub struct DefaultGraphAdapter {
    config_adapter: DefaultConfigAdapter,
}

impl DefaultGraphAdapter {
    pub fn new() -> Self {
        Self {
            config_adapter: DefaultConfigAdapter,
        }
    }
}

impl Default for DefaultGraphAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphAdapter for DefaultGraphAdapter {
    fn convert(&self, graph: &BuildGraph) -> AphelionResult<RacComputeGraph> {
        self.validate(graph)?;

        let mut rac_graph = RacComputeGraph::default();

        for node in &graph.nodes {
            let rac_config = self.config_adapter.convert(&node.config)?;
            let handle = RacNodeHandle::from(node.id);

            rac_graph.nodes.push(RacGraphNode {
                handle,
                op_type: node.name.clone(),
                config: rac_config,
                input_shapes: Vec::new(),
                output_shapes: Vec::new(),
            });
        }

        for (from, to) in &graph.edges {
            rac_graph
                .edges
                .push((RacNodeHandle::from(*from), RacNodeHandle::from(*to)));
        }

        rac_graph.metadata = RacGraphMetadata {
            source_framework: "aphelion-core".to_string(),
            content_hash: graph.stable_hash(),
            is_optimized: false,
            device_hints: Vec::new(),
        };

        Ok(rac_graph)
    }

    fn validate(&self, graph: &BuildGraph) -> AphelionResult<()> {
        let node_ids: std::collections::HashSet<_> = graph.nodes.iter().map(|n| n.id).collect();

        for (from, to) in &graph.edges {
            if !node_ids.contains(from) {
                return Err(AphelionError::graph(format!(
                    "Edge references non-existent source node: {:?}",
                    from
                )));
            }
            if !node_ids.contains(to) {
                return Err(AphelionError::graph(format!(
                    "Edge references non-existent target node: {:?}",
                    to
                )));
            }
        }

        Ok(())
    }
}

/// Placeholder runtime for testing.
#[derive(Debug, Clone, Default)]
pub struct PlaceholderRuntime {
    available_devices: Vec<RacDevice>,
}

impl PlaceholderRuntime {
    pub fn new() -> Self {
        Self {
            available_devices: vec![RacDevice::default_cpu()],
        }
    }

    pub fn with_device(mut self, device: RacDevice) -> Self {
        self.available_devices.push(device);
        self
    }
}

/// Placeholder output for runtime execution.
#[derive(Debug, Clone)]
pub struct PlaceholderOutput {
    pub success: bool,
    pub execution_time_ms: u64,
    pub device_used: String,
    pub nodes_executed: usize,
}

impl RuntimeAdapter for PlaceholderRuntime {
    type Output = PlaceholderOutput;

    fn execute(
        &self,
        graph: &RacComputeGraph,
        device: &RacDevice,
    ) -> AphelionResult<Self::Output> {
        if !self.is_device_available(device) {
            return Err(AphelionError::backend(format!(
                "Device not available: {}",
                device.id
            )));
        }

        Ok(PlaceholderOutput {
            success: true,
            execution_time_ms: graph.nodes.len() as u64 * 10,
            device_used: device.id.clone(),
            nodes_executed: graph.nodes.len(),
        })
    }

    fn is_device_available(&self, device: &RacDevice) -> bool {
        self.available_devices.iter().any(|d| d.id == device.id)
    }

    fn available_devices(&self) -> Vec<RacDevice> {
        self.available_devices.clone()
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Convert graph and config to rust-ai-core format.
pub fn to_rust_ai_core(
    graph: &BuildGraph,
    config: &ModelConfig,
) -> AphelionResult<(RacComputeGraph, RacModelConfig)> {
    let graph_adapter = DefaultGraphAdapter::new();
    let config_adapter = DefaultConfigAdapter;

    let rac_graph = graph_adapter.convert(graph)?;
    let rac_config = config_adapter.convert(config)?;

    Ok((rac_graph, rac_config))
}

/// Convert graph to rust-ai-core format.
pub fn graph_to_rac(graph: &BuildGraph) -> AphelionResult<RacComputeGraph> {
    DefaultGraphAdapter::new().convert(graph)
}

/// Convert config to rust-ai-core format.
pub fn config_to_rac(config: &ModelConfig) -> AphelionResult<RacModelConfig> {
    DefaultConfigAdapter.convert(config)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::graph::BuildGraph;

    #[test]
    fn test_rac_device_creation() {
        let cpu = RacDevice::default_cpu();
        assert_eq!(cpu.device_type, RacDeviceType::Cpu);
        assert_eq!(cpu.id, "cpu:0");

        let cuda = RacDevice::cuda(0).with_memory(8 * 1024 * 1024 * 1024);
        assert_eq!(cuda.device_type, RacDeviceType::Cuda);
        assert_eq!(cuda.id, "cuda:0");
        assert_eq!(cuda.memory_bytes, Some(8 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_config_adapter() {
        let config = ModelConfig::new("bert", "1.0.0")
            .with_param("batch_size", serde_json::json!(32))
            .with_param("hidden_size", serde_json::json!(768))
            .with_param("dtype", serde_json::json!("float16"));

        let adapter = DefaultConfigAdapter;
        let rac_config = adapter.convert(&config).unwrap();

        assert_eq!(rac_config.name, "bert");
        assert_eq!(rac_config.batch_size, Some(32));
        assert_eq!(rac_config.hidden_size, Some(768));
        assert_eq!(rac_config.dtype, RacDataType::Float16);
    }

    #[test]
    fn test_dtype_parsing() {
        assert_eq!(parse_dtype("float32").unwrap(), RacDataType::Float32);
        assert_eq!(parse_dtype("bf16").unwrap(), RacDataType::BFloat16);
        assert!(parse_dtype("invalid").is_err());
    }

    #[test]
    fn test_graph_adapter() {
        let mut graph = BuildGraph::default();
        let node1 = graph.add_node("encoder", ModelConfig::new("enc", "1.0"));
        let node2 = graph.add_node("decoder", ModelConfig::new("dec", "1.0"));
        graph.add_edge(node1, node2);

        let adapter = DefaultGraphAdapter::new();
        let rac_graph = adapter.convert(&graph).unwrap();

        assert_eq!(rac_graph.nodes.len(), 2);
        assert_eq!(rac_graph.edges.len(), 1);
        assert_eq!(rac_graph.metadata.content_hash, graph.stable_hash());
    }

    #[test]
    fn test_placeholder_runtime() {
        let runtime = PlaceholderRuntime::new().with_device(RacDevice::cuda(0));
        assert_eq!(runtime.available_devices().len(), 2);
        assert!(runtime.is_device_available(&RacDevice::default_cpu()));
    }

    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::with_limit(1024);
        assert!(tracker.would_fit(512));
        tracker.allocate(512).unwrap();
        assert_eq!(tracker.allocated_bytes(), 512);
        tracker.deallocate(256);
        assert_eq!(tracker.allocated_bytes(), 256);
    }

    #[test]
    fn test_aphelion_device() {
        let device = AphelionDevice::cpu();
        assert!(device.is_cpu());
        assert!(!device.is_cuda());
    }
}
