//! # rust-ai-core Integration Module
//!
//! This module provides adapter traits and implementations for integrating
//! aphelion-core with the hypothetical `rust-ai-core` framework.
//!
//! ## Overview
//!
//! The rust-ai-core adapter bridges aphelion-core's graph-based model building
//! system with rust-ai-core's execution runtime. This enables:
//!
//! - **Configuration Translation**: Converting [`ModelConfig`] to rust-ai-core's
//!   native configuration format
//! - **Graph Mapping**: Transforming [`BuildGraph`] into rust-ai-core's computation
//!   graph representation
//! - **Backend Bridging**: Connecting aphelion's backend abstraction with
//!   rust-ai-core's device management
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────┐     ┌──────────────────────┐
//! │   aphelion-core     │     │    rust-ai-core      │
//! ├─────────────────────┤     ├──────────────────────┤
//! │  ModelConfig        │────▶│  RacModelConfig      │
//! │  BuildGraph         │────▶│  RacComputeGraph     │
//! │  Backend            │────▶│  RacDevice           │
//! │  NodeId             │────▶│  RacNodeHandle       │
//! └─────────────────────┘     └──────────────────────┘
//! ```
//!
//! ## Feature Gate
//!
//! This module requires the `rust-ai-core` feature to be enabled:
//!
//! ```toml
//! [dependencies]
//! aphelion-core = { version = "1.0", features = ["rust-ai-core"] }
//! ```
//!
//! ## Real Integration Requirements
//!
//! When `rust-ai-core` becomes available as an actual dependency, the following
//! changes would be required:
//!
//! 1. Add `rust-ai-core` to `Cargo.toml` under the feature flag
//! 2. Replace placeholder types with actual `rust-ai-core` types
//! 3. Implement proper error mapping from rust-ai-core errors
//! 4. Add runtime device discovery and capability detection
//! 5. Implement async execution support if rust-ai-core provides it
//!
//! ## Example Usage
//!
//! ```ignore
//! use aphelion_core::rust_ai_core::adapter::*;
//! use aphelion_core::config::ModelConfig;
//! use aphelion_core::graph::BuildGraph;
//!
//! let config = ModelConfig::new("my-model", "1.0.0");
//! let graph = BuildGraph::default();
//!
//! // Convert to rust-ai-core types
//! let rac_config = RacModelConfig::from_aphelion(&config)?;
//! let rac_graph = RacComputeGraph::from_aphelion(&graph)?;
//!
//! // Execute on rust-ai-core runtime
//! let device = RacDevice::default_cpu();
//! let result = rac_graph.execute(&device, &rac_config)?;
//! ```

use crate::config::ModelConfig;
use crate::error::{AphelionError, AphelionResult};
use crate::graph::{BuildGraph, NodeId};

// ============================================================================
// Placeholder Types (representing rust-ai-core equivalents)
// ============================================================================

/// Placeholder for rust-ai-core's device abstraction.
///
/// In a real integration, this would be imported from `rust_ai_core::Device`
/// and would provide GPU/TPU/NPU device management capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RacDevice {
    /// Device identifier (e.g., "cuda:0", "cpu", "tpu:0")
    pub id: String,
    /// Device type classification
    pub device_type: RacDeviceType,
    /// Memory capacity in bytes (if applicable)
    pub memory_bytes: Option<u64>,
}

/// Device type classification for rust-ai-core.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RacDeviceType {
    /// CPU execution
    Cpu,
    /// NVIDIA CUDA GPU
    Cuda,
    /// AMD ROCm GPU
    Rocm,
    /// Intel oneAPI
    OneApi,
    /// Apple Metal
    Metal,
    /// Vulkan compute
    Vulkan,
    /// Custom/unknown accelerator
    Custom,
}

impl RacDevice {
    /// Create a CPU device instance.
    pub fn default_cpu() -> Self {
        Self {
            id: "cpu:0".to_string(),
            device_type: RacDeviceType::Cpu,
            memory_bytes: None,
        }
    }

    /// Create a CUDA device instance.
    pub fn cuda(index: u32) -> Self {
        Self {
            id: format!("cuda:{}", index),
            device_type: RacDeviceType::Cuda,
            memory_bytes: None,
        }
    }

    /// Create a device with specified memory capacity.
    pub fn with_memory(mut self, bytes: u64) -> Self {
        self.memory_bytes = Some(bytes);
        self
    }
}

/// Placeholder for rust-ai-core's model configuration format.
///
/// This represents the target configuration schema that rust-ai-core
/// would expect. Real integration would map aphelion's flexible
/// `BTreeMap<String, Value>` params to strongly-typed fields.
#[derive(Debug, Clone, PartialEq)]
pub struct RacModelConfig {
    /// Model name identifier
    pub name: String,
    /// Semantic version string
    pub version: String,
    /// Batch size for inference/training
    pub batch_size: Option<u32>,
    /// Sequence length for transformer models
    pub sequence_length: Option<u32>,
    /// Hidden dimension size
    pub hidden_size: Option<u32>,
    /// Number of attention heads (transformer models)
    pub num_attention_heads: Option<u32>,
    /// Number of layers
    pub num_layers: Option<u32>,
    /// Vocabulary size
    pub vocab_size: Option<u32>,
    /// Data type for computations
    pub dtype: RacDataType,
    /// Additional custom parameters
    pub custom_params: std::collections::BTreeMap<String, String>,
}

/// Data type for rust-ai-core computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RacDataType {
    /// 32-bit floating point
    #[default]
    Float32,
    /// 16-bit floating point
    Float16,
    /// Brain floating point (16-bit)
    BFloat16,
    /// 64-bit floating point
    Float64,
    /// 32-bit integer
    Int32,
    /// 64-bit integer
    Int64,
    /// 8-bit integer (quantized)
    Int8,
    /// 8-bit unsigned integer
    UInt8,
}

impl Default for RacModelConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            version: "0.0.0".to_string(),
            batch_size: None,
            sequence_length: None,
            hidden_size: None,
            num_attention_heads: None,
            num_layers: None,
            vocab_size: None,
            dtype: RacDataType::default(),
            custom_params: std::collections::BTreeMap::new(),
        }
    }
}

/// Placeholder for rust-ai-core's node handle in a compute graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RacNodeHandle {
    /// Internal graph index
    pub index: u64,
    /// Generation counter for validity checking
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

/// Placeholder for rust-ai-core's compute graph representation.
///
/// This would be the target graph format that rust-ai-core uses
/// for optimization and execution scheduling.
#[derive(Debug, Clone, Default)]
pub struct RacComputeGraph {
    /// Nodes in topological order
    pub nodes: Vec<RacGraphNode>,
    /// Edges as (source, target) pairs
    pub edges: Vec<(RacNodeHandle, RacNodeHandle)>,
    /// Graph-level metadata
    pub metadata: RacGraphMetadata,
}

/// A node in the rust-ai-core compute graph.
#[derive(Debug, Clone)]
pub struct RacGraphNode {
    /// Node handle for referencing
    pub handle: RacNodeHandle,
    /// Operation type
    pub op_type: String,
    /// Node-specific configuration
    pub config: RacModelConfig,
    /// Input tensor shapes (if known statically)
    pub input_shapes: Vec<Vec<i64>>,
    /// Output tensor shapes (if known statically)
    pub output_shapes: Vec<Vec<i64>>,
}

/// Metadata for a rust-ai-core compute graph.
#[derive(Debug, Clone, Default)]
pub struct RacGraphMetadata {
    /// Original source framework
    pub source_framework: String,
    /// Hash for reproducibility
    pub content_hash: String,
    /// Whether the graph has been optimized
    pub is_optimized: bool,
    /// Target device hints
    pub device_hints: Vec<String>,
}

// ============================================================================
// Adapter Traits
// ============================================================================

/// Trait for converting aphelion-core configurations to rust-ai-core format.
///
/// This trait defines the contract for configuration translation. Implementors
/// should handle the mapping of aphelion's flexible parameter system to
/// rust-ai-core's structured configuration.
///
/// # Example
///
/// ```
/// use aphelion_core::rust_ai_core::{ConfigAdapter, RacModelConfig, DefaultConfigAdapter};
/// use aphelion_core::config::ModelConfig;
///
/// let aphelion_config = ModelConfig::new("bert-base", "1.0.0")
///     .with_param("batch_size", serde_json::json!(32))
///     .with_param("hidden_size", serde_json::json!(768));
///
/// let adapter = DefaultConfigAdapter;
/// let rac_config = adapter.convert(&aphelion_config).unwrap();
///
/// assert_eq!(rac_config.name, "bert-base");
/// assert_eq!(rac_config.batch_size, Some(32));
/// ```
pub trait ConfigAdapter: Send + Sync {
    /// Convert an aphelion ModelConfig to rust-ai-core's RacModelConfig.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration contains invalid or
    /// incompatible parameters.
    fn convert(&self, config: &ModelConfig) -> AphelionResult<RacModelConfig>;

    /// Validate that a configuration is compatible with rust-ai-core.
    ///
    /// This can be used for early validation before attempting conversion.
    fn validate(&self, config: &ModelConfig) -> AphelionResult<()> {
        // Default implementation just attempts conversion
        self.convert(config).map(|_| ())
    }

    /// Get the list of supported parameter keys.
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

/// Trait for converting aphelion-core graphs to rust-ai-core format.
///
/// This trait handles the transformation of aphelion's `BuildGraph` into
/// rust-ai-core's `RacComputeGraph`, including node mapping and edge
/// translation.
///
/// # Graph Conversion Process
///
/// 1. Iterate through aphelion nodes in order
/// 2. Convert each node's configuration using `ConfigAdapter`
/// 3. Map NodeId to RacNodeHandle
/// 4. Preserve edge connectivity
/// 5. Compute graph metadata (hash, source info)
///
/// # Example
///
/// ```
/// use aphelion_core::rust_ai_core::{GraphAdapter, DefaultGraphAdapter};
/// use aphelion_core::graph::BuildGraph;
/// use aphelion_core::config::ModelConfig;
///
/// let mut graph = BuildGraph::default();
/// let node1 = graph.add_node("encoder", ModelConfig::new("encoder", "1.0"));
/// let node2 = graph.add_node("decoder", ModelConfig::new("decoder", "1.0"));
/// graph.add_edge(node1, node2);
///
/// let adapter = DefaultGraphAdapter::new();
/// let rac_graph = adapter.convert(&graph).unwrap();
///
/// assert_eq!(rac_graph.nodes.len(), 2);
/// assert_eq!(rac_graph.edges.len(), 1);
/// ```
pub trait GraphAdapter: Send + Sync {
    /// Convert an aphelion BuildGraph to rust-ai-core's RacComputeGraph.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph structure is invalid or contains
    /// nodes with incompatible configurations.
    fn convert(&self, graph: &BuildGraph) -> AphelionResult<RacComputeGraph>;

    /// Convert a graph with a specific target device hint.
    fn convert_for_device(
        &self,
        graph: &BuildGraph,
        device: &RacDevice,
    ) -> AphelionResult<RacComputeGraph> {
        let mut rac_graph = self.convert(graph)?;
        rac_graph.metadata.device_hints.push(device.id.clone());
        Ok(rac_graph)
    }

    /// Validate graph structure for rust-ai-core compatibility.
    fn validate(&self, graph: &BuildGraph) -> AphelionResult<()>;
}

/// Trait for executing converted graphs on rust-ai-core runtime.
///
/// This trait would be implemented when actual rust-ai-core runtime
/// is available. It provides the execution interface for running
/// converted graphs on various devices.
pub trait RuntimeAdapter: Send + Sync {
    /// The output type produced by execution.
    type Output;

    /// Execute a compute graph on the specified device.
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails due to device issues,
    /// memory constraints, or runtime errors.
    fn execute(
        &self,
        graph: &RacComputeGraph,
        device: &RacDevice,
    ) -> AphelionResult<Self::Output>;

    /// Check if a device is available for execution.
    fn is_device_available(&self, device: &RacDevice) -> bool;

    /// List all available devices.
    fn available_devices(&self) -> Vec<RacDevice>;
}

// ============================================================================
// Default Implementations
// ============================================================================

/// Default configuration adapter with standard parameter mapping.
#[derive(Debug, Clone, Default)]
pub struct DefaultConfigAdapter;

impl ConfigAdapter for DefaultConfigAdapter {
    fn convert(&self, config: &ModelConfig) -> AphelionResult<RacModelConfig> {
        let mut rac_config = RacModelConfig {
            name: config.name.clone(),
            version: config.version.clone(),
            ..Default::default()
        };

        // Extract known parameters
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

        // Store remaining parameters as custom
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

/// Parse a dtype string into RacDataType.
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
        _ => Err(AphelionError::config_error(format!(
            "Unknown dtype: {}",
            s
        ))),
    }
}

/// Default graph adapter implementation.
#[derive(Debug, Clone)]
pub struct DefaultGraphAdapter {
    config_adapter: DefaultConfigAdapter,
}

impl DefaultGraphAdapter {
    /// Create a new default graph adapter.
    pub fn new() -> Self {
        Self {
            config_adapter: DefaultConfigAdapter,
        }
    }

    /// Create with a custom config adapter.
    pub fn with_config_adapter(config_adapter: DefaultConfigAdapter) -> Self {
        Self { config_adapter }
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

        // Convert nodes
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

        // Convert edges
        for (from, to) in &graph.edges {
            rac_graph.edges.push((
                RacNodeHandle::from(*from),
                RacNodeHandle::from(*to),
            ));
        }

        // Set metadata
        rac_graph.metadata = RacGraphMetadata {
            source_framework: "aphelion-core".to_string(),
            content_hash: graph.stable_hash(),
            is_optimized: false,
            device_hints: Vec::new(),
        };

        Ok(rac_graph)
    }

    fn validate(&self, graph: &BuildGraph) -> AphelionResult<()> {
        // Check for cycles (simple check - real implementation would use proper cycle detection)
        // This is a placeholder validation

        // Check that all edge references are valid
        let node_ids: std::collections::HashSet<_> =
            graph.nodes.iter().map(|n| n.id).collect();

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

/// Placeholder runtime adapter for testing.
///
/// This implementation simulates rust-ai-core runtime behavior
/// without actual computation capabilities.
#[derive(Debug, Clone, Default)]
pub struct PlaceholderRuntime {
    available_devices: Vec<RacDevice>,
}

impl PlaceholderRuntime {
    /// Create a new placeholder runtime with default CPU device.
    pub fn new() -> Self {
        Self {
            available_devices: vec![RacDevice::default_cpu()],
        }
    }

    /// Add an available device.
    pub fn with_device(mut self, device: RacDevice) -> Self {
        self.available_devices.push(device);
        self
    }
}

/// Placeholder output type for runtime execution.
#[derive(Debug, Clone)]
pub struct PlaceholderOutput {
    /// Whether execution completed successfully
    pub success: bool,
    /// Execution time in milliseconds (simulated)
    pub execution_time_ms: u64,
    /// Device used for execution
    pub device_used: String,
    /// Number of nodes executed
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

        // Simulate execution
        Ok(PlaceholderOutput {
            success: true,
            execution_time_ms: graph.nodes.len() as u64 * 10, // Simulated timing
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

/// Convert an aphelion graph and config to rust-ai-core format.
///
/// This is a convenience function that uses the default adapters.
///
/// # Example
///
/// ```
/// use aphelion_core::rust_ai_core::to_rust_ai_core;
/// use aphelion_core::graph::BuildGraph;
/// use aphelion_core::config::ModelConfig;
///
/// let graph = BuildGraph::default();
/// let config = ModelConfig::new("test", "1.0");
///
/// let (rac_graph, rac_config) = to_rust_ai_core(&graph, &config).unwrap();
/// ```
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

/// Create a rust-ai-core compute graph from an aphelion graph.
pub fn graph_to_rac(graph: &BuildGraph) -> AphelionResult<RacComputeGraph> {
    DefaultGraphAdapter::new().convert(graph)
}

/// Create a rust-ai-core config from an aphelion config.
pub fn config_to_rac(config: &ModelConfig) -> AphelionResult<RacModelConfig> {
    DefaultConfigAdapter.convert(config)
}

// ============================================================================
// Feature-gated actual adapter (when rust-ai-core is available)
// ============================================================================

/// Module containing the actual rust-ai-core integration.
///
/// This module is only compiled when the `rust-ai-core` feature is enabled
/// AND the actual rust-ai-core crate is available as a dependency.
///
/// ## Future Implementation Notes
///
/// When rust-ai-core becomes available:
///
/// 1. Import actual types: `use rust_ai_core::{Device, Graph, Config};`
/// 2. Implement `From<ModelConfig>` for actual rust-ai-core Config
/// 3. Implement `From<BuildGraph>` for actual rust-ai-core Graph
/// 4. Add proper error type conversions
/// 5. Implement async execution if supported
#[cfg(feature = "rust-ai-core")]
pub mod adapter {
    pub use super::*;

    // When rust-ai-core is available, this would contain:
    //
    // ```ignore
    // use rust_ai_core as rac;
    //
    // impl From<&ModelConfig> for rac::Config {
    //     fn from(config: &ModelConfig) -> Self {
    //         // Real conversion logic
    //     }
    // }
    //
    // impl From<&BuildGraph> for rac::Graph {
    //     fn from(graph: &BuildGraph) -> Self {
    //         // Real conversion logic
    //     }
    // }
    // ```
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
        assert!(cpu.memory_bytes.is_none());

        let cuda = RacDevice::cuda(0).with_memory(8 * 1024 * 1024 * 1024);
        assert_eq!(cuda.device_type, RacDeviceType::Cuda);
        assert_eq!(cuda.id, "cuda:0");
        assert_eq!(cuda.memory_bytes, Some(8 * 1024 * 1024 * 1024));
    }

    #[test]
    fn test_config_adapter_basic() {
        let config = ModelConfig::new("test-model", "1.0.0");
        let adapter = DefaultConfigAdapter;

        let rac_config = adapter.convert(&config).unwrap();
        assert_eq!(rac_config.name, "test-model");
        assert_eq!(rac_config.version, "1.0.0");
    }

    #[test]
    fn test_config_adapter_with_params() {
        let config = ModelConfig::new("bert", "1.0.0")
            .with_param("batch_size", serde_json::json!(32))
            .with_param("hidden_size", serde_json::json!(768))
            .with_param("num_attention_heads", serde_json::json!(12))
            .with_param("num_layers", serde_json::json!(12))
            .with_param("vocab_size", serde_json::json!(30522))
            .with_param("dtype", serde_json::json!("float16"));

        let adapter = DefaultConfigAdapter;
        let rac_config = adapter.convert(&config).unwrap();

        assert_eq!(rac_config.batch_size, Some(32));
        assert_eq!(rac_config.hidden_size, Some(768));
        assert_eq!(rac_config.num_attention_heads, Some(12));
        assert_eq!(rac_config.num_layers, Some(12));
        assert_eq!(rac_config.vocab_size, Some(30522));
        assert_eq!(rac_config.dtype, RacDataType::Float16);
    }

    #[test]
    fn test_config_adapter_custom_params() {
        let config = ModelConfig::new("custom", "1.0.0")
            .with_param("custom_key", serde_json::json!("custom_value"))
            .with_param("another_param", serde_json::json!(42));

        let adapter = DefaultConfigAdapter;
        let rac_config = adapter.convert(&config).unwrap();

        assert!(rac_config.custom_params.contains_key("custom_key"));
        assert!(rac_config.custom_params.contains_key("another_param"));
    }

    #[test]
    fn test_dtype_parsing() {
        assert_eq!(parse_dtype("float32").unwrap(), RacDataType::Float32);
        assert_eq!(parse_dtype("f32").unwrap(), RacDataType::Float32);
        assert_eq!(parse_dtype("FLOAT16").unwrap(), RacDataType::Float16);
        assert_eq!(parse_dtype("bfloat16").unwrap(), RacDataType::BFloat16);
        assert_eq!(parse_dtype("bf16").unwrap(), RacDataType::BFloat16);

        assert!(parse_dtype("invalid").is_err());
    }

    #[test]
    fn test_graph_adapter_empty() {
        let graph = BuildGraph::default();
        let adapter = DefaultGraphAdapter::new();

        let rac_graph = adapter.convert(&graph).unwrap();
        assert!(rac_graph.nodes.is_empty());
        assert!(rac_graph.edges.is_empty());
        assert_eq!(rac_graph.metadata.source_framework, "aphelion-core");
    }

    #[test]
    fn test_graph_adapter_with_nodes() {
        let mut graph = BuildGraph::default();
        let node1 = graph.add_node("encoder", ModelConfig::new("enc", "1.0"));
        let node2 = graph.add_node("decoder", ModelConfig::new("dec", "1.0"));
        graph.add_edge(node1, node2);

        let adapter = DefaultGraphAdapter::new();
        let rac_graph = adapter.convert(&graph).unwrap();

        assert_eq!(rac_graph.nodes.len(), 2);
        assert_eq!(rac_graph.edges.len(), 1);
        assert_eq!(rac_graph.nodes[0].op_type, "encoder");
        assert_eq!(rac_graph.nodes[1].op_type, "decoder");
    }

    #[test]
    fn test_graph_adapter_preserves_hash() {
        let mut graph = BuildGraph::default();
        graph.add_node("test", ModelConfig::new("test", "1.0"));

        let adapter = DefaultGraphAdapter::new();
        let rac_graph = adapter.convert(&graph).unwrap();

        assert_eq!(rac_graph.metadata.content_hash, graph.stable_hash());
    }

    #[test]
    fn test_graph_validation_invalid_edge() {
        let mut graph = BuildGraph::default();
        let node1 = graph.add_node("test", ModelConfig::new("test", "1.0"));
        // Add edge to non-existent node
        graph.edges.push((node1, NodeId::new(999)));

        let adapter = DefaultGraphAdapter::new();
        let result = adapter.validate(&graph);

        assert!(result.is_err());
    }

    #[test]
    fn test_node_id_to_handle_conversion() {
        let node_id = NodeId::new(42);
        let handle: RacNodeHandle = node_id.into();

        assert_eq!(handle.index, 42);
        assert_eq!(handle.generation, 0);
    }

    #[test]
    fn test_placeholder_runtime() {
        let runtime = PlaceholderRuntime::new()
            .with_device(RacDevice::cuda(0));

        let devices = runtime.available_devices();
        assert_eq!(devices.len(), 2); // CPU + CUDA

        assert!(runtime.is_device_available(&RacDevice::default_cpu()));
        assert!(runtime.is_device_available(&RacDevice::cuda(0)));
        assert!(!runtime.is_device_available(&RacDevice::cuda(1)));
    }

    #[test]
    fn test_placeholder_runtime_execute() {
        let mut graph = BuildGraph::default();
        graph.add_node("op1", ModelConfig::new("op1", "1.0"));
        graph.add_node("op2", ModelConfig::new("op2", "1.0"));

        let adapter = DefaultGraphAdapter::new();
        let rac_graph = adapter.convert(&graph).unwrap();

        let runtime = PlaceholderRuntime::new();
        let device = RacDevice::default_cpu();

        let output = runtime.execute(&rac_graph, &device).unwrap();
        assert!(output.success);
        assert_eq!(output.nodes_executed, 2);
        assert_eq!(output.device_used, "cpu:0");
    }

    #[test]
    fn test_placeholder_runtime_unavailable_device() {
        let rac_graph = RacComputeGraph::default();
        let runtime = PlaceholderRuntime::new(); // Only has CPU

        let result = runtime.execute(&rac_graph, &RacDevice::cuda(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_convenience_functions() {
        let mut graph = BuildGraph::default();
        graph.add_node("test", ModelConfig::new("test", "1.0"));
        let config = ModelConfig::new("model", "2.0");

        let (rac_graph, rac_config) = to_rust_ai_core(&graph, &config).unwrap();
        assert_eq!(rac_graph.nodes.len(), 1);
        assert_eq!(rac_config.name, "model");

        let rac_graph2 = graph_to_rac(&graph).unwrap();
        assert_eq!(rac_graph2.nodes.len(), 1);

        let rac_config2 = config_to_rac(&config).unwrap();
        assert_eq!(rac_config2.version, "2.0");
    }

    #[test]
    fn test_convert_for_device() {
        let mut graph = BuildGraph::default();
        graph.add_node("test", ModelConfig::new("test", "1.0"));

        let adapter = DefaultGraphAdapter::new();
        let device = RacDevice::cuda(0);

        let rac_graph = adapter.convert_for_device(&graph, &device).unwrap();
        assert!(rac_graph.metadata.device_hints.contains(&"cuda:0".to_string()));
    }

    #[test]
    fn test_supported_params() {
        let adapter = DefaultConfigAdapter;
        let params = adapter.supported_params();

        assert!(params.contains(&"batch_size"));
        assert!(params.contains(&"hidden_size"));
        assert!(params.contains(&"dtype"));
    }

    #[test]
    fn test_rac_model_config_default() {
        let config = RacModelConfig::default();

        assert!(config.name.is_empty());
        assert_eq!(config.version, "0.0.0");
        assert!(config.batch_size.is_none());
        assert_eq!(config.dtype, RacDataType::Float32);
    }

    #[test]
    fn test_complex_graph_conversion() {
        // Create a more complex graph structure
        let mut graph = BuildGraph::default();

        let embed = graph.add_node(
            "embedding",
            ModelConfig::new("embed", "1.0")
                .with_param("vocab_size", serde_json::json!(30000))
                .with_param("hidden_size", serde_json::json!(512)),
        );

        let enc1 = graph.add_node(
            "encoder_layer_1",
            ModelConfig::new("encoder", "1.0")
                .with_param("num_attention_heads", serde_json::json!(8)),
        );

        let enc2 = graph.add_node(
            "encoder_layer_2",
            ModelConfig::new("encoder", "1.0")
                .with_param("num_attention_heads", serde_json::json!(8)),
        );

        let output = graph.add_node(
            "output_projection",
            ModelConfig::new("linear", "1.0"),
        );

        graph.add_edge(embed, enc1);
        graph.add_edge(enc1, enc2);
        graph.add_edge(enc2, output);

        let adapter = DefaultGraphAdapter::new();
        let rac_graph = adapter.convert(&graph).unwrap();

        assert_eq!(rac_graph.nodes.len(), 4);
        assert_eq!(rac_graph.edges.len(), 3);

        // Verify node configs were converted
        assert_eq!(rac_graph.nodes[0].config.vocab_size, Some(30000));
        assert_eq!(rac_graph.nodes[1].config.num_attention_heads, Some(8));
    }
}
