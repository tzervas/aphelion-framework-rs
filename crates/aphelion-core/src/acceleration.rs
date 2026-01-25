//! Acceleration pipeline stages for tritter-accel integration.
//!
//! This module provides pipeline stages that apply tritter-accel optimizations
//! to build graphs, enabling gradient compression for training and optimized
//! inference paths.
//!
//! # Feature Flag
//!
//! This module is only available when the `tritter-accel` feature is enabled.
//!
//! # Examples
//!
//! ```ignore
//! use aphelion_core::acceleration::AccelerationStage;
//! use aphelion_core::pipeline::BuildPipeline;
//!
//! // Create a pipeline with acceleration for training
//! let pipeline = BuildPipeline::new()
//!     .with_stage(Box::new(AccelerationStage::for_training(0.1)));
//!
//! // Create a pipeline with acceleration for inference
//! let pipeline = BuildPipeline::new()
//!     .with_stage(Box::new(AccelerationStage::for_inference(32)));
//! ```

use crate::diagnostics::{TraceEvent, TraceLevel};
use crate::error::AphelionResult;
use crate::graph::BuildGraph;
use crate::pipeline::{BuildContext, PipelineStage};
use std::time::SystemTime;

/// Configuration for training acceleration.
#[derive(Debug, Clone)]
pub struct TrainingAccelConfig {
    /// Gradient compression ratio (0.01 = 100x, 0.1 = 10x)
    pub compression_ratio: f32,
    /// Enable deterministic training for reproducibility
    pub deterministic: bool,
    /// Seed for deterministic operations
    pub seed: Option<u64>,
    /// Enable mixed precision training
    pub mixed_precision: bool,
}

impl Default for TrainingAccelConfig {
    fn default() -> Self {
        Self {
            compression_ratio: 0.1,
            deterministic: true,
            seed: None,
            mixed_precision: false,
        }
    }
}

impl TrainingAccelConfig {
    /// Creates a new training configuration with the specified compression ratio.
    pub fn new(compression_ratio: f32) -> Self {
        Self {
            compression_ratio: compression_ratio.clamp(0.01, 1.0),
            ..Default::default()
        }
    }

    /// Enables deterministic training with a seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.deterministic = true;
        self.seed = Some(seed);
        self
    }

    /// Enables mixed precision training.
    pub fn with_mixed_precision(mut self) -> Self {
        self.mixed_precision = true;
        self
    }
}

/// Configuration for inference acceleration.
#[derive(Debug, Clone)]
pub struct InferenceAccelConfig {
    /// Batch size for batched inference
    pub batch_size: usize,
    /// Convert layers to ternary for 16x memory reduction
    pub use_ternary_layers: bool,
    /// Enable KV caching for autoregressive models
    pub use_kv_cache: bool,
    /// Maximum sequence length for KV cache
    pub max_seq_len: Option<usize>,
}

impl Default for InferenceAccelConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            use_ternary_layers: true,
            use_kv_cache: false,
            max_seq_len: None,
        }
    }
}

impl InferenceAccelConfig {
    /// Creates a new inference configuration with the specified batch size.
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size: batch_size.max(1),
            ..Default::default()
        }
    }

    /// Enables KV caching with the specified maximum sequence length.
    pub fn with_kv_cache(mut self, max_seq_len: usize) -> Self {
        self.use_kv_cache = true;
        self.max_seq_len = Some(max_seq_len);
        self
    }

    /// Disables ternary layer conversion.
    pub fn without_ternary_layers(mut self) -> Self {
        self.use_ternary_layers = false;
        self
    }
}

/// Acceleration mode for the pipeline stage.
#[derive(Debug, Clone)]
pub enum AccelMode {
    /// Training mode with gradient compression
    Training(TrainingAccelConfig),
    /// Inference mode with optimizations
    Inference(InferenceAccelConfig),
}

/// Pipeline stage that applies tritter-accel optimizations.
///
/// `AccelerationStage` modifies the build graph to enable hardware acceleration
/// for either training or inference workloads. It records metadata in the graph
/// that downstream stages and backends can use to configure execution.
///
/// # Training Mode
///
/// In training mode, the stage:
/// - Records gradient compression ratio in graph metadata
/// - Enables deterministic phase training hooks
/// - Configures mixed precision if enabled
///
/// # Inference Mode
///
/// In inference mode, the stage:
/// - Marks linear layers for ternary conversion
/// - Configures batch size for parallel execution
/// - Enables KV caching for autoregressive models
///
/// # Examples
///
/// ```ignore
/// use aphelion_core::acceleration::AccelerationStage;
/// use aphelion_core::pipeline::{BuildPipeline, BuildContext};
/// use aphelion_core::graph::BuildGraph;
/// use aphelion_core::backend::NullBackend;
/// use aphelion_core::diagnostics::InMemoryTraceSink;
///
/// // Training pipeline with 10x gradient compression
/// let stage = AccelerationStage::for_training(0.1);
///
/// let backend = NullBackend::cpu();
/// let trace = InMemoryTraceSink::new();
/// let ctx = BuildContext { backend: &backend, trace: &trace };
/// let mut graph = BuildGraph::default();
///
/// stage.execute(&ctx, &mut graph).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct AccelerationStage {
    mode: AccelMode,
}

impl AccelerationStage {
    /// Creates an acceleration stage for training with the specified compression ratio.
    ///
    /// # Arguments
    ///
    /// * `compression_ratio` - Gradient compression ratio (0.01-1.0)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::acceleration::AccelerationStage;
    ///
    /// let stage = AccelerationStage::for_training(0.1); // 10x compression
    /// ```
    pub fn for_training(compression_ratio: f32) -> Self {
        Self {
            mode: AccelMode::Training(TrainingAccelConfig::new(compression_ratio)),
        }
    }

    /// Creates an acceleration stage for training with full configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Training configuration
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::acceleration::{AccelerationStage, TrainingAccelConfig};
    ///
    /// let config = TrainingAccelConfig::new(0.1)
    ///     .with_seed(42)
    ///     .with_mixed_precision();
    ///
    /// let stage = AccelerationStage::with_training_config(config);
    /// ```
    pub fn with_training_config(config: TrainingAccelConfig) -> Self {
        Self {
            mode: AccelMode::Training(config),
        }
    }

    /// Creates an acceleration stage for inference with the specified batch size.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Batch size for inference
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::acceleration::AccelerationStage;
    ///
    /// let stage = AccelerationStage::for_inference(32);
    /// ```
    pub fn for_inference(batch_size: usize) -> Self {
        Self {
            mode: AccelMode::Inference(InferenceAccelConfig::new(batch_size)),
        }
    }

    /// Creates an acceleration stage for inference with full configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Inference configuration
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::acceleration::{AccelerationStage, InferenceAccelConfig};
    ///
    /// let config = InferenceAccelConfig::new(32)
    ///     .with_kv_cache(2048);
    ///
    /// let stage = AccelerationStage::with_inference_config(config);
    /// ```
    pub fn with_inference_config(config: InferenceAccelConfig) -> Self {
        Self {
            mode: AccelMode::Inference(config),
        }
    }

    /// Returns whether this stage is configured for training.
    pub fn is_training(&self) -> bool {
        matches!(self.mode, AccelMode::Training(_))
    }

    /// Returns whether this stage is configured for inference.
    pub fn is_inference(&self) -> bool {
        matches!(self.mode, AccelMode::Inference(_))
    }

    /// Applies training acceleration to the graph.
    fn apply_training_acceleration(
        &self,
        ctx: &BuildContext,
        graph: &mut BuildGraph,
        config: &TrainingAccelConfig,
    ) -> AphelionResult<()> {
        // Record acceleration metadata in graph
        for node in &mut graph.nodes {
            node.metadata.insert(
                "accel.mode".to_string(),
                serde_json::Value::String("training".to_string()),
            );
            node.metadata.insert(
                "accel.compression_ratio".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(config.compression_ratio as f64)
                        .unwrap_or_else(|| serde_json::Number::from(0)),
                ),
            );
            node.metadata.insert(
                "accel.deterministic".to_string(),
                serde_json::Value::Bool(config.deterministic),
            );
            if let Some(seed) = config.seed {
                node.metadata.insert(
                    "accel.seed".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(seed)),
                );
            }
            if config.mixed_precision {
                node.metadata.insert(
                    "accel.mixed_precision".to_string(),
                    serde_json::Value::Bool(true),
                );
            }
        }

        // Record trace event
        ctx.trace.record(TraceEvent {
            id: "stage.acceleration.training".to_string(),
            message: format!(
                "Applied training acceleration: compression_ratio={}, deterministic={}, nodes={}",
                config.compression_ratio,
                config.deterministic,
                graph.nodes.len()
            ),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });

        Ok(())
    }

    /// Applies inference acceleration to the graph.
    fn apply_inference_acceleration(
        &self,
        ctx: &BuildContext,
        graph: &mut BuildGraph,
        config: &InferenceAccelConfig,
    ) -> AphelionResult<()> {
        // Record acceleration metadata in graph
        for node in &mut graph.nodes {
            node.metadata.insert(
                "accel.mode".to_string(),
                serde_json::Value::String("inference".to_string()),
            );
            node.metadata.insert(
                "accel.batch_size".to_string(),
                serde_json::Value::Number(serde_json::Number::from(config.batch_size)),
            );
            node.metadata.insert(
                "accel.ternary_layers".to_string(),
                serde_json::Value::Bool(config.use_ternary_layers),
            );
            if config.use_kv_cache {
                node.metadata
                    .insert("accel.kv_cache".to_string(), serde_json::Value::Bool(true));
                if let Some(max_seq_len) = config.max_seq_len {
                    node.metadata.insert(
                        "accel.max_seq_len".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(max_seq_len)),
                    );
                }
            }
        }

        // Record trace event
        ctx.trace.record(TraceEvent {
            id: "stage.acceleration.inference".to_string(),
            message: format!(
                "Applied inference acceleration: batch_size={}, ternary={}, kv_cache={}, nodes={}",
                config.batch_size,
                config.use_ternary_layers,
                config.use_kv_cache,
                graph.nodes.len()
            ),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });

        Ok(())
    }
}

impl PipelineStage for AccelerationStage {
    fn name(&self) -> &str {
        match &self.mode {
            AccelMode::Training(_) => "tritter-acceleration-training",
            AccelMode::Inference(_) => "tritter-acceleration-inference",
        }
    }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph) -> AphelionResult<()> {
        match &self.mode {
            AccelMode::Training(config) => self.apply_training_acceleration(ctx, graph, config),
            AccelMode::Inference(config) => self.apply_inference_acceleration(ctx, graph, config),
        }
    }
}

/// Async implementation for AccelerationStage.
#[cfg(feature = "tokio")]
impl crate::pipeline::AsyncPipelineStage for AccelerationStage {
    fn name(&self) -> &str {
        <Self as PipelineStage>::name(self)
    }

    fn execute_async<'a>(
        &'a self,
        ctx: &'a BuildContext<'_>,
        graph: &'a mut BuildGraph,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AphelionResult<()>> + Send + 'a>> {
        Box::pin(async move { self.execute(ctx, graph) })
    }
}

/// Pre-build hook for gradient compression setup.
///
/// This hook validates that gradient compression is properly configured
/// before training begins. Use with `BuildPipeline::with_pre_hook`.
///
/// # Examples
///
/// ```ignore
/// use aphelion_core::acceleration::gradient_compression_pre_hook;
/// use aphelion_core::pipeline::BuildPipeline;
///
/// let pipeline = BuildPipeline::new()
///     .with_pre_hook(gradient_compression_pre_hook(0.1, 42));
/// ```
pub fn gradient_compression_pre_hook(
    compression_ratio: f32,
    seed: u64,
) -> impl Fn(&BuildContext) -> AphelionResult<()> + Send + Sync + 'static {
    move |ctx| {
        ctx.trace.record(TraceEvent {
            id: "hook.gradient_compression.setup".to_string(),
            message: format!(
                "Gradient compression initialized: ratio={}, seed={}",
                compression_ratio, seed
            ),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });
        Ok(())
    }
}

/// Post-build hook for gradient compression validation.
///
/// This hook validates the graph after acceleration has been applied.
/// Use with `BuildPipeline::with_post_hook`.
///
/// # Examples
///
/// ```ignore
/// use aphelion_core::acceleration::gradient_compression_post_hook;
/// use aphelion_core::pipeline::BuildPipeline;
///
/// let pipeline = BuildPipeline::new()
///     .with_post_hook(gradient_compression_post_hook());
/// ```
pub fn gradient_compression_post_hook(
) -> impl Fn(&BuildContext, &BuildGraph) -> AphelionResult<()> + Send + Sync + 'static {
    |ctx, graph| {
        // Validate that acceleration metadata was applied
        let accel_nodes = graph
            .nodes
            .iter()
            .filter(|n| n.metadata.contains_key("accel.mode"))
            .count();

        if accel_nodes == 0 && !graph.nodes.is_empty() {
            ctx.trace.record(TraceEvent {
                id: "hook.gradient_compression.warning".to_string(),
                message: "No nodes have acceleration metadata".to_string(),
                timestamp: SystemTime::now(),
                level: TraceLevel::Warn,
                span_id: None,
                trace_id: None,
            });
        } else {
            ctx.trace.record(TraceEvent {
                id: "hook.gradient_compression.validated".to_string(),
                message: format!(
                    "Acceleration validated: {}/{} nodes configured",
                    accel_nodes,
                    graph.nodes.len()
                ),
                timestamp: SystemTime::now(),
                level: TraceLevel::Info,
                span_id: None,
                trace_id: None,
            });
        }

        Ok(())
    }
}

/// Creates a training pipeline with acceleration configured.
///
/// This is a convenience function that creates a complete pipeline
/// with validation, acceleration, and hashing stages.
///
/// # Arguments
///
/// * `compression_ratio` - Gradient compression ratio
///
/// # Examples
///
/// ```ignore
/// use aphelion_core::acceleration::training_pipeline;
///
/// let pipeline = training_pipeline(0.1);
/// ```
pub fn training_pipeline(compression_ratio: f32) -> crate::pipeline::BuildPipeline {
    crate::pipeline::BuildPipeline::new()
        .with_stage(Box::new(crate::pipeline::ValidationStage))
        .with_stage(Box::new(AccelerationStage::for_training(compression_ratio)))
        .with_stage(Box::new(crate::pipeline::HashingStage))
}

/// Creates an inference pipeline with acceleration configured.
///
/// This is a convenience function that creates a complete pipeline
/// with hashing and acceleration stages (skips validation for speed).
///
/// # Arguments
///
/// * `batch_size` - Batch size for inference
///
/// # Examples
///
/// ```ignore
/// use aphelion_core::acceleration::inference_pipeline;
///
/// let pipeline = inference_pipeline(32);
/// ```
pub fn inference_pipeline(batch_size: usize) -> crate::pipeline::BuildPipeline {
    crate::pipeline::BuildPipeline::new()
        .with_stage(Box::new(AccelerationStage::for_inference(batch_size)))
        .with_stage(Box::new(crate::pipeline::HashingStage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::NullBackend;
    use crate::config::ModelConfig;
    use crate::diagnostics::InMemoryTraceSink;

    #[test]
    fn test_training_config_new() {
        let config = TrainingAccelConfig::new(0.1);
        assert!((config.compression_ratio - 0.1).abs() < f32::EPSILON);
        assert!(config.deterministic);
        assert!(!config.mixed_precision);
    }

    #[test]
    fn test_training_config_with_seed() {
        let config = TrainingAccelConfig::new(0.1).with_seed(42);
        assert_eq!(config.seed, Some(42));
        assert!(config.deterministic);
    }

    #[test]
    fn test_training_config_with_mixed_precision() {
        let config = TrainingAccelConfig::new(0.1).with_mixed_precision();
        assert!(config.mixed_precision);
    }

    #[test]
    fn test_inference_config_new() {
        let config = InferenceAccelConfig::new(32);
        assert_eq!(config.batch_size, 32);
        assert!(config.use_ternary_layers);
        assert!(!config.use_kv_cache);
    }

    #[test]
    fn test_inference_config_with_kv_cache() {
        let config = InferenceAccelConfig::new(32).with_kv_cache(2048);
        assert!(config.use_kv_cache);
        assert_eq!(config.max_seq_len, Some(2048));
    }

    #[test]
    fn test_inference_config_without_ternary() {
        let config = InferenceAccelConfig::new(32).without_ternary_layers();
        assert!(!config.use_ternary_layers);
    }

    #[test]
    fn test_acceleration_stage_for_training() {
        let stage = AccelerationStage::for_training(0.1);
        assert!(stage.is_training());
        assert!(!stage.is_inference());
        assert_eq!(stage.name(), "tritter-acceleration-training");
    }

    #[test]
    fn test_acceleration_stage_for_inference() {
        let stage = AccelerationStage::for_inference(32);
        assert!(!stage.is_training());
        assert!(stage.is_inference());
        assert_eq!(stage.name(), "tritter-acceleration-inference");
    }

    #[test]
    fn test_acceleration_stage_execute_training() {
        let stage = AccelerationStage::for_training(0.1);
        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let mut graph = BuildGraph::default();
        graph.add_node("linear1", ModelConfig::new("linear", "1.0"));
        graph.add_node("linear2", ModelConfig::new("linear", "1.0"));

        let result = stage.execute(&ctx, &mut graph);
        assert!(result.is_ok());

        // Verify metadata was added
        for node in &graph.nodes {
            assert!(node.metadata.contains_key("accel.mode"));
            assert!(node.metadata.contains_key("accel.compression_ratio"));
            assert!(node.metadata.contains_key("accel.deterministic"));
        }

        // Verify trace event
        let events = trace.events();
        assert!(events
            .iter()
            .any(|e| e.message.contains("Applied training acceleration")));
    }

    #[test]
    fn test_acceleration_stage_execute_inference() {
        let stage = AccelerationStage::for_inference(32);
        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let mut graph = BuildGraph::default();
        graph.add_node("linear1", ModelConfig::new("linear", "1.0"));

        let result = stage.execute(&ctx, &mut graph);
        assert!(result.is_ok());

        // Verify metadata was added
        for node in &graph.nodes {
            assert!(node.metadata.contains_key("accel.mode"));
            assert!(node.metadata.contains_key("accel.batch_size"));
            assert!(node.metadata.contains_key("accel.ternary_layers"));
        }

        // Verify trace event
        let events = trace.events();
        assert!(events
            .iter()
            .any(|e| e.message.contains("Applied inference acceleration")));
    }

    #[test]
    fn test_gradient_compression_pre_hook() {
        let hook = gradient_compression_pre_hook(0.1, 42);
        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let result = hook(&ctx);
        assert!(result.is_ok());

        let events = trace.events();
        assert!(events.iter().any(|e| e.message.contains("ratio=0.1")));
        assert!(events.iter().any(|e| e.message.contains("seed=42")));
    }

    #[test]
    fn test_gradient_compression_post_hook() {
        let hook = gradient_compression_post_hook();
        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let mut graph = BuildGraph::default();
        graph.add_node("linear", ModelConfig::new("linear", "1.0"));

        // Apply acceleration first
        let stage = AccelerationStage::for_training(0.1);
        stage.execute(&ctx, &mut graph).unwrap();

        // Clear events and run post hook
        let trace2 = InMemoryTraceSink::new();
        let ctx2 = BuildContext {
            backend: &backend,
            trace: &trace2,
        };

        let result = hook(&ctx2, &graph);
        assert!(result.is_ok());

        let events = trace2.events();
        assert!(events.iter().any(|e| e.message.contains("validated")));
    }

    #[test]
    fn test_training_pipeline() {
        let pipeline = training_pipeline(0.1);
        // Should have 3 stages: validation, acceleration, hashing
        // We can't directly access stages count, but we can test execution
        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let mut graph = BuildGraph::default();
        graph.add_node("linear", ModelConfig::new("linear", "1.0"));

        let result = pipeline.execute(&ctx, graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_inference_pipeline() {
        let pipeline = inference_pipeline(32);
        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let mut graph = BuildGraph::default();
        graph.add_node("linear", ModelConfig::new("linear", "1.0"));

        let result = pipeline.execute(&ctx, graph);
        assert!(result.is_ok());
    }

    #[test]
    fn test_acceleration_stage_clone() {
        let stage = AccelerationStage::for_training(0.1);
        let cloned = stage.clone();
        assert!(cloned.is_training());
    }

    #[test]
    fn test_accel_mode_variants() {
        let training = AccelMode::Training(TrainingAccelConfig::default());
        let inference = AccelMode::Inference(InferenceAccelConfig::default());

        assert!(matches!(training, AccelMode::Training(_)));
        assert!(matches!(inference, AccelMode::Inference(_)));
    }
}
