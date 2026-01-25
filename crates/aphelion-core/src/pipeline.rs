//! Build pipeline orchestration and stage management.
//!
//! This module provides the infrastructure for defining and executing multi-stage
//! build pipelines with support for hooks, progress tracking, and stage skipping.
//! The pipeline architecture enables composable, reusable build processes.

use crate::backend::{Backend, ModelBuilder};
use crate::config::ConfigSpec;
use crate::diagnostics::{TraceEvent, TraceLevel, TraceSink, TraceSinkExt};
use crate::error::{AphelionError, AphelionResult};
use crate::graph::BuildGraph;
use std::collections::HashSet;
use std::time::SystemTime;

#[cfg(feature = "rust-ai-core")]
use crate::rust_ai_core::MemoryTracker;

/// Type alias for pre-build hook functions.
pub type PreBuildHook = Box<dyn Fn(&BuildContext) -> AphelionResult<()> + Send + Sync>;

/// Type alias for post-build hook functions.
pub type PostBuildHook =
    Box<dyn Fn(&BuildContext, &BuildGraph) -> AphelionResult<()> + Send + Sync>;

/// Type alias for progress callback functions.
pub type ProgressCallback = Box<dyn Fn(&str, usize, usize) + Send + Sync>;

/// Execution context containing backend and tracing infrastructure.
///
/// `BuildContext` provides the runtime environment for pipeline stages and builders.
/// It supplies both the computational backend and the tracing sink for recording events.
///
/// # Fields
///
/// * `backend` - The computational backend to use for operations
/// * `trace` - The trace sink for recording diagnostic events
/// * `memory_tracker` - Optional memory tracker (requires `rust-ai-core` feature)
///
/// # Examples
///
/// ```
/// use aphelion_core::pipeline::BuildContext;
/// use aphelion_core::backend::NullBackend;
/// use aphelion_core::diagnostics::InMemoryTraceSink;
///
/// let backend = NullBackend::cpu();
/// let trace = InMemoryTraceSink::new();
///
/// let ctx = BuildContext::new(&backend, &trace);
/// ```
pub struct BuildContext<'a> {
    /// The computational backend
    pub backend: &'a dyn Backend,
    /// The trace sink for recording events
    pub trace: &'a dyn TraceSink,
    /// Optional memory tracker for GPU memory management
    #[cfg(feature = "rust-ai-core")]
    pub memory_tracker: Option<&'a MemoryTracker>,
}

impl<'a> BuildContext<'a> {
    /// Creates a new `BuildContext` with the provided backend and trace sink.
    #[cfg(feature = "rust-ai-core")]
    pub fn new(backend: &'a dyn Backend, trace: &'a dyn TraceSink) -> Self {
        Self {
            backend,
            trace,
            memory_tracker: None,
        }
    }

    /// Creates a new `BuildContext` with the provided backend and trace sink.
    #[cfg(not(feature = "rust-ai-core"))]
    pub fn new(backend: &'a dyn Backend, trace: &'a dyn TraceSink) -> Self {
        Self { backend, trace }
    }

    /// Creates a new `BuildContext` with a provided null backend for testing scenarios.
    #[cfg(feature = "rust-ai-core")]
    pub fn with_null_backend(
        backend: &'a crate::backend::NullBackend,
        trace: &'a dyn TraceSink,
    ) -> Self {
        Self {
            backend,
            trace,
            memory_tracker: None,
        }
    }

    /// Creates a new `BuildContext` with a provided null backend for testing scenarios.
    #[cfg(not(feature = "rust-ai-core"))]
    pub fn with_null_backend(
        backend: &'a crate::backend::NullBackend,
        trace: &'a dyn TraceSink,
    ) -> Self {
        Self { backend, trace }
    }

    /// Creates a new `BuildContext` with memory tracking enabled.
    ///
    /// The memory tracker can be used to monitor GPU memory usage during
    /// pipeline execution and prevent OOM errors.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use aphelion_core::pipeline::BuildContext;
    /// use aphelion_core::rust_ai_core::MemoryTracker;
    ///
    /// let tracker = MemoryTracker::with_limit(8 * 1024 * 1024 * 1024); // 8GB
    /// let ctx = BuildContext::with_memory_tracker(&backend, &trace, &tracker);
    /// ```
    #[cfg(feature = "rust-ai-core")]
    pub fn with_memory_tracker(
        backend: &'a dyn Backend,
        trace: &'a dyn TraceSink,
        memory_tracker: &'a MemoryTracker,
    ) -> Self {
        Self {
            backend,
            trace,
            memory_tracker: Some(memory_tracker),
        }
    }

    /// Check if an allocation would fit in the memory budget.
    ///
    /// Returns `true` if no memory tracker is configured or if the
    /// allocation would fit within the limit.
    #[cfg(feature = "rust-ai-core")]
    pub fn would_fit(&self, bytes: usize) -> bool {
        self.memory_tracker
            .map(|t| t.would_fit(bytes))
            .unwrap_or(true)
    }

    /// Record an allocation in the memory tracker.
    ///
    /// Returns `Ok(())` if no memory tracker is configured or if the
    /// allocation succeeds. Returns an error if the allocation would
    /// exceed the memory limit.
    #[cfg(feature = "rust-ai-core")]
    pub fn allocate(&self, bytes: usize) -> AphelionResult<()> {
        if let Some(tracker) = self.memory_tracker {
            tracker
                .allocate(bytes)
                .map_err(|e| AphelionError::backend(format!("OOM: {}", e)))?;
        }
        Ok(())
    }

    /// Record a deallocation in the memory tracker.
    #[cfg(feature = "rust-ai-core")]
    pub fn deallocate(&self, bytes: usize) {
        if let Some(tracker) = self.memory_tracker {
            tracker.deallocate(bytes);
        }
    }

    /// Get the current allocated bytes from the memory tracker.
    ///
    /// Returns `None` if no memory tracker is configured.
    #[cfg(feature = "rust-ai-core")]
    pub fn allocated_bytes(&self) -> Option<usize> {
        self.memory_tracker.map(|t| t.allocated_bytes())
    }

    /// Get the peak allocated bytes from the memory tracker.
    ///
    /// Returns `None` if no memory tracker is configured.
    #[cfg(feature = "rust-ai-core")]
    pub fn peak_bytes(&self) -> Option<usize> {
        self.memory_tracker.map(|t| t.peak_bytes())
    }
}

/// Trait for composable pipeline stages.
///
/// `PipelineStage` defines the interface for individual stages in a build pipeline.
/// Stages are executed sequentially, each receiving the output of the previous stage,
/// enabling composable, modular build processes.
///
/// # Implementing PipelineStage
///
/// Types implementing `PipelineStage` must be thread-safe (`Send + Sync`) and
/// deterministic - running the same stage with the same inputs should produce
/// the same outputs.
///
/// # Examples
///
/// ```
/// use aphelion_core::pipeline::{PipelineStage, BuildContext};
/// use aphelion_core::graph::BuildGraph;
/// use aphelion_core::error::AphelionResult;
///
/// struct LoggingStage;
///
/// impl PipelineStage for LoggingStage {
///     fn name(&self) -> &str {
///         "logging"
///     }
///
///     fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph) -> AphelionResult<()> {
///         // Implementation here
///         Ok(())
///     }
/// }
/// ```
pub trait PipelineStage: Send + Sync {
    /// Returns the name of this stage for logging and identification.
    ///
    /// Stage names are used in progress reporting and error messages, so they
    /// should be descriptive and unique within a pipeline.
    fn name(&self) -> &str;

    /// Executes this stage with the given context and graph.
    ///
    /// The stage receives a mutable reference to the graph, allowing it to
    /// modify the graph in-place (adding nodes, edges, metadata, etc.).
    ///
    /// # Arguments
    ///
    /// * `ctx` - The build context with backend and trace sink
    /// * `graph` - The computation graph being built
    ///
    /// # Errors
    ///
    /// Returns `AphelionResult::Err` if the stage fails.
    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph) -> AphelionResult<()>;
}

/// Trait for composable asynchronous pipeline stages.
///
/// `AsyncPipelineStage` defines the interface for asynchronous stages in a build pipeline.
/// This trait is only available when the `tokio` feature is enabled.
///
/// # Implementing AsyncPipelineStage
///
/// Types implementing `AsyncPipelineStage` must be thread-safe (`Send + Sync`) and
/// deterministic. The async execution should have the same behavior as sync stages.
///
/// # Examples
///
/// ```ignore
/// use aphelion_core::pipeline::{AsyncPipelineStage, BuildContext};
/// use aphelion_core::graph::BuildGraph;
/// use aphelion_core::error::AphelionResult;
/// use std::pin::Pin;
/// use std::future::Future;
///
/// struct AsyncLoggingStage;
///
/// impl AsyncPipelineStage for AsyncLoggingStage {
///     fn name(&self) -> &str {
///         "async_logging"
///     }
///
///     fn execute_async<'a>(
///         &'a self,
///         ctx: &'a BuildContext<'_>,
///         graph: &'a mut BuildGraph,
///     ) -> Pin<Box<dyn Future<Output = AphelionResult<()>> + Send + 'a>> {
///         Box::pin(async move {
///             // Async implementation here
///             Ok(())
///         })
///     }
/// }
/// ```
#[cfg(feature = "tokio")]
pub trait AsyncPipelineStage: Send + Sync {
    /// Returns the name of this stage for logging and identification.
    fn name(&self) -> &str;

    /// Executes this stage asynchronously with the given context and graph.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The build context with backend and trace sink
    /// * `graph` - The computation graph being built
    ///
    /// # Errors
    ///
    /// Returns `AphelionResult::Err` if the stage fails.
    fn execute_async<'a>(
        &'a self,
        ctx: &'a BuildContext<'_>,
        graph: &'a mut BuildGraph,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AphelionResult<()>> + Send + 'a>>;
}

/// Lifecycle hooks for pipeline execution.
///
/// `PipelineHooks` allows registering callbacks that execute before and after
/// the main pipeline stages. Pre-build hooks execute before any stages, and
/// post-build hooks execute after all stages complete successfully.
///
/// # Fields
///
/// * `pre_build` - Callbacks to execute before stages
/// * `post_build` - Callbacks to execute after stages
///
/// # Examples
///
/// ```
/// use aphelion_core::pipeline::PipelineHooks;
/// use aphelion_core::pipeline::BuildContext;
/// use aphelion_core::backend::NullBackend;
/// use aphelion_core::diagnostics::InMemoryTraceSink;
///
/// let mut hooks = PipelineHooks::default();
/// // Can add hooks here
/// ```
#[derive(Default)]
pub struct PipelineHooks {
    /// Callbacks to execute before any pipeline stages
    pub pre_build: Vec<PreBuildHook>,
    /// Callbacks to execute after all pipeline stages
    pub post_build: Vec<PostBuildHook>,
}

/// An extensible build pipeline for composing model construction stages.
///
/// `BuildPipeline` orchestrates the execution of multiple stages, hooks, and progress callbacks.
/// Stages are executed in order, with support for selective stage skipping and progress reporting.
///
/// # Examples
///
/// ```
/// use aphelion_core::pipeline::{BuildPipeline, ValidationStage, HashingStage};
/// use aphelion_core::backend::NullBackend;
/// use aphelion_core::diagnostics::InMemoryTraceSink;
/// use aphelion_core::pipeline::BuildContext;
/// use aphelion_core::graph::BuildGraph;
/// use aphelion_core::config::ModelConfig;
///
/// let mut graph = BuildGraph::default();
/// graph.add_node("test", ModelConfig::new("model", "1.0.0"));
///
/// let pipeline = BuildPipeline::new()
///     .with_stage(Box::new(ValidationStage))
///     .with_stage(Box::new(HashingStage))
///     .with_progress(|name, current, total| {
///         println!("Stage {} ({}/{})", name, current, total);
///     });
///
/// let backend = NullBackend::cpu();
/// let trace = InMemoryTraceSink::new();
/// let ctx = BuildContext::new(&backend, &trace);
///
/// let result = pipeline.execute(&ctx, graph);
/// assert!(result.is_ok());
/// ```
pub struct BuildPipeline {
    stages: Vec<Box<dyn PipelineStage>>,
    #[cfg(feature = "tokio")]
    async_stages: Vec<Box<dyn AsyncPipelineStage>>,
    hooks: PipelineHooks,
    skip_stages: HashSet<String>,
    on_progress: Option<ProgressCallback>,
}

impl Default for BuildPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl BuildPipeline {
    /// Creates a new empty pipeline.
    ///
    /// The pipeline starts with no stages, hooks, or progress callbacks.
    /// Use the builder methods to configure it.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::pipeline::BuildPipeline;
    ///
    /// let pipeline = BuildPipeline::new();
    /// // Configure pipeline...
    /// ```
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            #[cfg(feature = "tokio")]
            async_stages: Vec::new(),
            hooks: PipelineHooks::default(),
            skip_stages: HashSet::new(),
            on_progress: None,
        }
    }

    /// Creates a standard pipeline with validation and hashing stages.
    ///
    /// The standard pipeline provides a sensible default for most use cases:
    /// - Validates the build graph structure
    /// - Computes and traces the deterministic graph hash
    ///
    /// This is the recommended pipeline for general model building.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::pipeline::BuildPipeline;
    ///
    /// let pipeline = BuildPipeline::standard();
    /// // Use pipeline for building...
    /// ```
    pub fn standard() -> Self {
        Self::new()
            .with_stage(Box::new(ValidationStage))
            .with_stage(Box::new(HashingStage))
    }

    /// Creates a training pipeline optimized for model training workflows.
    ///
    /// The training pipeline extends the standard pipeline with:
    /// - All standard stages (validation + hashing)
    /// - Pre-hook that logs the start of training
    ///
    /// This pipeline is useful for workflows that need to distinguish
    /// between training and inference execution phases.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::pipeline::BuildPipeline;
    ///
    /// let pipeline = BuildPipeline::for_training();
    /// // Use pipeline for training...
    /// ```
    pub fn for_training() -> Self {
        Self::standard().with_pre_hook(|ctx| {
            ctx.trace
                .info("pipeline.training", "Starting training pipeline");
            Ok(())
        })
    }

    /// Creates an inference pipeline optimized for deployment and inference.
    ///
    /// The inference pipeline is minimal for speed:
    /// - Only computes and traces the graph hash
    /// - Skips expensive validation for known-good models
    ///
    /// This pipeline is ideal for production deployments where the model
    /// has already been validated during training.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::pipeline::BuildPipeline;
    ///
    /// let pipeline = BuildPipeline::for_inference();
    /// // Use pipeline for inference...
    /// ```
    pub fn for_inference() -> Self {
        Self::new().with_stage(Box::new(HashingStage))
    }

    /// Adds a stage to the pipeline.
    ///
    /// Stages are executed in the order they are added.
    ///
    /// # Arguments
    ///
    /// * `stage` - The stage to add
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::pipeline::{BuildPipeline, ValidationStage};
    ///
    /// let pipeline = BuildPipeline::new()
    ///     .with_stage(Box::new(ValidationStage));
    /// ```
    pub fn with_stage(mut self, stage: Box<dyn PipelineStage>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Adds an async stage to the pipeline.
    ///
    /// Async stages are only available when the `tokio` feature is enabled.
    /// They are executed in the order they are added.
    ///
    /// # Arguments
    ///
    /// * `stage` - The async stage to add
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::pipeline::BuildPipeline;
    ///
    /// let pipeline = BuildPipeline::new()
    ///     .with_async_stage(Box::new(MyAsyncStage));
    /// ```
    #[cfg(feature = "tokio")]
    pub fn with_async_stage(mut self, stage: Box<dyn AsyncPipelineStage>) -> Self {
        self.async_stages.push(stage);
        self
    }

    /// Adds a pre-build hook.
    ///
    /// Pre-build hooks execute before any stages. If a hook returns an error,
    /// the pipeline stops execution.
    ///
    /// # Arguments
    ///
    /// * `hook` - Closure to execute before stages
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::pipeline::BuildPipeline;
    ///
    /// let pipeline = BuildPipeline::new()
    ///     .with_pre_hook(|ctx| {
    ///         println!("Starting build");
    ///         Ok(())
    ///     });
    /// ```
    pub fn with_pre_hook<F>(mut self, hook: F) -> Self
    where
        F: Fn(&BuildContext) -> AphelionResult<()> + Send + Sync + 'static,
    {
        self.hooks.pre_build.push(Box::new(hook));
        self
    }

    /// Adds a post-build hook.
    ///
    /// Post-build hooks execute after all stages complete successfully.
    ///
    /// # Arguments
    ///
    /// * `hook` - Closure to execute after stages
    pub fn with_post_hook<F>(mut self, hook: F) -> Self
    where
        F: Fn(&BuildContext, &BuildGraph) -> AphelionResult<()> + Send + Sync + 'static,
    {
        self.hooks.post_build.push(Box::new(hook));
        self
    }

    /// Skips a stage by name during execution.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the stage to skip
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::pipeline::BuildPipeline;
    ///
    /// let pipeline = BuildPipeline::new()
    ///     .with_skip_stage("validation");
    /// ```
    pub fn with_skip_stage(mut self, name: impl Into<String>) -> Self {
        self.skip_stages.insert(name.into());
        self
    }

    /// Sets a progress callback for monitoring pipeline execution.
    ///
    /// The callback is invoked for each stage with the stage name, current position,
    /// and total number of stages. Skipped stages are still reported.
    ///
    /// # Arguments
    ///
    /// * `callback` - Closure receiving (stage_name, current_position, total_stages)
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::pipeline::BuildPipeline;
    ///
    /// let pipeline = BuildPipeline::new()
    ///     .with_progress(|name, current, total| {
    ///         println!("Progress: {}/{} - {}", current, total, name);
    ///     });
    /// ```
    pub fn with_progress<F>(mut self, callback: F) -> Self
    where
        F: Fn(&str, usize, usize) + Send + Sync + 'static,
    {
        self.on_progress = Some(Box::new(callback));
        self
    }

    /// Executes the pipeline with all configured stages.
    ///
    /// Executes pre-build hooks, then all stages in order (skipping as configured),
    /// then post-build hooks. Returns the final graph if successful.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The build context
    /// * `graph` - The initial computation graph
    ///
    /// # Returns
    ///
    /// The final graph after all stages, or an error if any stage fails
    ///
    /// # Errors
    ///
    /// Returns `AphelionResult::Err` if any hook or stage returns an error.
    pub fn execute(
        &self,
        ctx: &BuildContext<'_>,
        mut graph: BuildGraph,
    ) -> AphelionResult<BuildGraph> {
        // Execute pre-build hooks
        for hook in &self.hooks.pre_build {
            hook(ctx)?;
        }

        // Execute stages
        let total_stages = self.stages.len();
        for (index, stage) in self.stages.iter().enumerate() {
            let stage_name = stage.name();

            // Skip stage if it's in the skip list
            if self.skip_stages.contains(stage_name) {
                if let Some(ref progress) = self.on_progress {
                    progress(
                        &format!("{} (skipped)", stage_name),
                        index + 1,
                        total_stages,
                    );
                }
                continue;
            }

            // Report progress
            if let Some(ref progress) = self.on_progress {
                progress(stage_name, index + 1, total_stages);
            }

            // Execute stage
            stage.execute(ctx, &mut graph)?;
        }

        // Execute post-build hooks
        for hook in &self.hooks.post_build {
            hook(ctx, &graph)?;
        }

        Ok(graph)
    }

    /// Executes the pipeline asynchronously with all configured async stages.
    ///
    /// This method is only available when the `tokio` feature is enabled.
    /// Executes pre-build hooks, then all async stages in order (skipping as configured),
    /// then post-build hooks. Returns the final graph if successful.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The build context
    /// * `graph` - The initial computation graph
    ///
    /// # Returns
    ///
    /// The final graph after all stages, or an error if any stage fails
    ///
    /// # Errors
    ///
    /// Returns `AphelionResult::Err` if any hook or stage returns an error.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use aphelion_core::pipeline::{BuildPipeline, BuildContext};
    /// use aphelion_core::graph::BuildGraph;
    /// use aphelion_core::backend::NullBackend;
    /// use aphelion_core::diagnostics::InMemoryTraceSink;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let pipeline = BuildPipeline::new();
    ///     let backend = NullBackend::cpu();
    ///     let trace = InMemoryTraceSink::new();
    ///     let ctx = BuildContext { backend: &backend, trace: &trace };
    ///     let graph = BuildGraph::default();
    ///
    ///     let result = pipeline.execute_async(&ctx, graph).await;
    ///     assert!(result.is_ok());
    /// }
    /// ```
    #[cfg(feature = "tokio")]
    pub async fn execute_async(
        &self,
        ctx: &BuildContext<'_>,
        mut graph: BuildGraph,
    ) -> AphelionResult<BuildGraph> {
        // Execute pre-build hooks
        for hook in &self.hooks.pre_build {
            hook(ctx)?;
        }

        // Execute async stages
        let total_stages = self.async_stages.len();
        for (index, stage) in self.async_stages.iter().enumerate() {
            let stage_name = stage.name();

            // Skip stage if it's in the skip list
            if self.skip_stages.contains(stage_name) {
                if let Some(ref progress) = self.on_progress {
                    progress(
                        &format!("{} (skipped)", stage_name),
                        index + 1,
                        total_stages,
                    );
                }
                continue;
            }

            // Report progress
            if let Some(ref progress) = self.on_progress {
                progress(stage_name, index + 1, total_stages);
            }

            // Execute async stage
            stage.execute_async(ctx, &mut graph).await?;
        }

        // Execute post-build hooks
        for hook in &self.hooks.post_build {
            hook(ctx, &graph)?;
        }

        Ok(graph)
    }

    /// Builds a model using the pipeline.
    ///
    /// Delegates to the model builder's `build` method and logs lifecycle events.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to build
    /// * `ctx` - The build context
    ///
    /// # Returns
    ///
    /// The constructed build graph
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let graph = BuildPipeline::build(&my_model, ctx)?;
    /// ```
    pub fn build<M: ModelBuilder<Output = BuildGraph>>(
        model: &M,
        ctx: BuildContext<'_>,
    ) -> AphelionResult<BuildGraph> {
        ctx.trace.record(TraceEvent {
            id: "pipeline.start".to_string(),
            message: "build started".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });

        let graph = model.build(ctx.backend, ctx.trace);

        ctx.trace.record(TraceEvent {
            id: "pipeline.finish".to_string(),
            message: format!("build completed hash={}", graph.stable_hash()),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });

        Ok(graph)
    }

    /// Builds a model after validating its configuration.
    ///
    /// Validates that the model's configuration has non-empty name and version before building.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to build (must implement ConfigSpec)
    /// * `ctx` - The build context
    ///
    /// # Errors
    ///
    /// Returns `AphelionError::InvalidConfig` if the configuration is invalid.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let graph = BuildPipeline::build_with_validation(&my_model, ctx)?;
    /// ```
    pub fn build_with_validation<M>(model: &M, ctx: BuildContext<'_>) -> AphelionResult<BuildGraph>
    where
        M: ModelBuilder<Output = BuildGraph> + ConfigSpec,
    {
        let config = model.config();
        if config.name.trim().is_empty() {
            return Err(AphelionError::config_error("name cannot be empty"));
        }
        if config.version.trim().is_empty() {
            return Err(AphelionError::config_error("version cannot be empty"));
        }

        ctx.trace.record(TraceEvent {
            id: "pipeline.validate".to_string(),
            message: format!("validated {}@{}", config.name, config.version),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });

        Self::build(model, ctx)
    }
}

/// A pipeline stage that validates the build graph structure.
///
/// `ValidationStage` ensures that the graph has at least one node, preventing
/// errors from empty graphs.
///
/// # Examples
///
/// ```
/// use aphelion_core::pipeline::{ValidationStage, PipelineStage, BuildContext};
/// use aphelion_core::backend::NullBackend;
/// use aphelion_core::diagnostics::InMemoryTraceSink;
/// use aphelion_core::graph::BuildGraph;
/// use aphelion_core::config::ModelConfig;
///
/// let mut graph = BuildGraph::default();
/// graph.add_node("test", ModelConfig::new("model", "1.0.0"));
///
/// let stage = ValidationStage;
/// let backend = NullBackend::cpu();
/// let trace = InMemoryTraceSink::new();
/// let ctx = BuildContext::new(&backend, &trace);
///
/// assert!(stage.execute(&ctx, &mut graph).is_ok());
/// ```
pub struct ValidationStage;

impl PipelineStage for ValidationStage {
    fn name(&self) -> &str {
        "validation"
    }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph) -> AphelionResult<()> {
        if graph.nodes.is_empty() {
            return Err(AphelionError::validation(
                "graph must contain at least one node",
            ));
        }

        ctx.trace.record(TraceEvent {
            id: "stage.validation".to_string(),
            message: format!("validated {} nodes", graph.nodes.len()),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });

        Ok(())
    }
}

/// A pipeline stage that computes and traces the graph's stable hash.
///
/// `HashingStage` computes a deterministic hash of the graph for traceability,
/// caching, and reproducibility verification.
///
/// # Examples
///
/// ```
/// use aphelion_core::pipeline::{HashingStage, PipelineStage, BuildContext};
/// use aphelion_core::backend::NullBackend;
/// use aphelion_core::diagnostics::InMemoryTraceSink;
/// use aphelion_core::graph::BuildGraph;
/// use aphelion_core::config::ModelConfig;
///
/// let mut graph = BuildGraph::default();
/// graph.add_node("test", ModelConfig::new("model", "1.0.0"));
///
/// let stage = HashingStage;
/// let backend = NullBackend::cpu();
/// let trace = InMemoryTraceSink::new();
/// let ctx = BuildContext::new(&backend, &trace);
///
/// assert!(stage.execute(&ctx, &mut graph).is_ok());
/// let events = trace.events();
/// assert!(events.iter().any(|e| e.message.contains("computed graph hash")));
/// ```
pub struct HashingStage;

impl PipelineStage for HashingStage {
    fn name(&self) -> &str {
        "hashing"
    }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph) -> AphelionResult<()> {
        let hash = graph.stable_hash();

        ctx.trace.record(TraceEvent {
            id: "stage.hashing".to_string(),
            message: format!("computed graph hash: {}", hash),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });

        Ok(())
    }
}

/// Async implementation for ValidationStage.
///
/// This allows ValidationStage to be used in async pipelines when the `tokio` feature is enabled.
///
/// # Note
///
/// This implementation delegates to the synchronous `execute()` method without yielding.
/// For fast operations like validation, this is acceptable. For CPU-intensive stages,
/// consider using `tokio::task::spawn_blocking()` in your custom implementations.
#[cfg(feature = "tokio")]
impl AsyncPipelineStage for ValidationStage {
    fn name(&self) -> &str {
        "validation"
    }

    fn execute_async<'a>(
        &'a self,
        ctx: &'a BuildContext<'_>,
        graph: &'a mut BuildGraph,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AphelionResult<()>> + Send + 'a>> {
        Box::pin(async move {
            // Delegate to the synchronous implementation
            // Note: This does not yield to the async runtime - acceptable for fast operations
            self.execute(ctx, graph)
        })
    }
}

/// Async implementation for HashingStage.
///
/// This allows HashingStage to be used in async pipelines when the `tokio` feature is enabled.
///
/// # Note
///
/// This implementation delegates to the synchronous `execute()` method without yielding.
/// For fast operations like hashing, this is acceptable. For CPU-intensive stages,
/// consider using `tokio::task::spawn_blocking()` in your custom implementations.
#[cfg(feature = "tokio")]
impl AsyncPipelineStage for HashingStage {
    fn name(&self) -> &str {
        "hashing"
    }

    fn execute_async<'a>(
        &'a self,
        ctx: &'a BuildContext<'_>,
        graph: &'a mut BuildGraph,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = AphelionResult<()>> + Send + 'a>> {
        Box::pin(async move {
            // Delegate to the synchronous implementation
            self.execute(ctx, graph)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::config::ModelConfig;
    use crate::diagnostics::TraceSink;
    use std::sync::{Arc, Mutex};

    /// Mock backend for testing
    struct MockBackend;
    impl Backend for MockBackend {
        fn name(&self) -> &str {
            "mock"
        }

        fn device(&self) -> &str {
            "mock_device"
        }

        fn capabilities(&self) -> crate::backend::DeviceCapabilities {
            crate::backend::DeviceCapabilities::default()
        }

        fn is_available(&self) -> bool {
            true
        }
    }

    /// Mock trace sink for testing
    struct MockTraceSink {
        events: Arc<Mutex<Vec<String>>>,
    }

    impl MockTraceSink {
        fn new() -> Self {
            Self {
                events: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn get_events(&self) -> Vec<String> {
            self.events.lock().unwrap().clone()
        }
    }

    impl TraceSink for MockTraceSink {
        fn record(&self, event: TraceEvent) {
            let mut events = self.events.lock().unwrap();
            events.push(format!("{}: {}", event.id, event.message));
        }
    }

    /// Test stage that records execution
    struct RecordingStage {
        name: String,
        executed: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingStage {
        fn new(name: &str, executed: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                name: name.to_string(),
                executed,
            }
        }
    }

    impl PipelineStage for RecordingStage {
        fn name(&self) -> &str {
            &self.name
        }

        fn execute(&self, _ctx: &BuildContext, _graph: &mut BuildGraph) -> AphelionResult<()> {
            self.executed.lock().unwrap().push(self.name.clone());
            Ok(())
        }
    }

    #[test]
    fn test_stage_execution_order() {
        let executed = Arc::new(Mutex::new(Vec::new()));

        let stage1 = Box::new(RecordingStage::new("stage1", Arc::clone(&executed)));
        let stage2 = Box::new(RecordingStage::new("stage2", Arc::clone(&executed)));
        let stage3 = Box::new(RecordingStage::new("stage3", Arc::clone(&executed)));

        let pipeline = BuildPipeline::new()
            .with_stage(stage1)
            .with_stage(stage2)
            .with_stage(stage3);

        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext::new(&MockBackend, &trace_sink);

        let graph = BuildGraph::default();
        let _result = pipeline.execute(&ctx, graph);

        let execution_order = executed.lock().unwrap().clone();
        assert_eq!(execution_order, vec!["stage1", "stage2", "stage3"]);
    }

    #[test]
    fn test_pre_hook_invocation() {
        let pre_hook_called = Arc::new(Mutex::new(false));
        let pre_hook_called_clone = Arc::clone(&pre_hook_called);

        let pipeline = BuildPipeline::new().with_pre_hook(move |_ctx| {
            *pre_hook_called_clone.lock().unwrap() = true;
            Ok(())
        });

        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext::new(&MockBackend, &trace_sink);

        let graph = BuildGraph::default();
        let _result = pipeline.execute(&ctx, graph);

        assert!(*pre_hook_called.lock().unwrap());
    }

    #[test]
    fn test_post_hook_invocation() {
        let post_hook_called = Arc::new(Mutex::new(false));
        let post_hook_called_clone = Arc::clone(&post_hook_called);

        let pipeline = BuildPipeline::new().with_post_hook(move |_ctx, _graph| {
            *post_hook_called_clone.lock().unwrap() = true;
            Ok(())
        });

        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext::new(&MockBackend, &trace_sink);

        let graph = BuildGraph::default();
        let _result = pipeline.execute(&ctx, graph);

        assert!(*post_hook_called.lock().unwrap());
    }

    #[test]
    fn test_stage_skipping() {
        let executed = Arc::new(Mutex::new(Vec::new()));

        let stage1 = Box::new(RecordingStage::new("stage1", Arc::clone(&executed)));
        let stage2 = Box::new(RecordingStage::new("stage2", Arc::clone(&executed)));
        let stage3 = Box::new(RecordingStage::new("stage3", Arc::clone(&executed)));

        let pipeline = BuildPipeline::new()
            .with_stage(stage1)
            .with_stage(stage2)
            .with_stage(stage3)
            .with_skip_stage("stage2");

        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext::new(&MockBackend, &trace_sink);

        let graph = BuildGraph::default();
        let _result = pipeline.execute(&ctx, graph);

        let execution_order = executed.lock().unwrap().clone();
        assert_eq!(execution_order, vec!["stage1", "stage3"]);
    }

    #[test]
    fn test_progress_callback() {
        let progress_calls = Arc::new(Mutex::new(Vec::new()));
        let progress_calls_clone = Arc::clone(&progress_calls);

        let stage1 = Box::new(RecordingStage::new(
            "stage1",
            Arc::new(Mutex::new(Vec::new())),
        ));
        let stage2 = Box::new(RecordingStage::new(
            "stage2",
            Arc::new(Mutex::new(Vec::new())),
        ));

        let pipeline = BuildPipeline::new()
            .with_stage(stage1)
            .with_stage(stage2)
            .with_progress(move |name, current, total| {
                progress_calls_clone
                    .lock()
                    .unwrap()
                    .push((name.to_string(), current, total));
            });

        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext::new(&MockBackend, &trace_sink);

        let graph = BuildGraph::default();
        let _result = pipeline.execute(&ctx, graph);

        let calls = progress_calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0], ("stage1".to_string(), 1, 2));
        assert_eq!(calls[1], ("stage2".to_string(), 2, 2));
    }

    #[test]
    fn test_progress_callback_with_skipped_stages() {
        let progress_calls = Arc::new(Mutex::new(Vec::new()));
        let progress_calls_clone = Arc::clone(&progress_calls);

        let stage1 = Box::new(RecordingStage::new(
            "stage1",
            Arc::new(Mutex::new(Vec::new())),
        ));
        let stage2 = Box::new(RecordingStage::new(
            "stage2",
            Arc::new(Mutex::new(Vec::new())),
        ));
        let stage3 = Box::new(RecordingStage::new(
            "stage3",
            Arc::new(Mutex::new(Vec::new())),
        ));

        let pipeline = BuildPipeline::new()
            .with_stage(stage1)
            .with_stage(stage2)
            .with_stage(stage3)
            .with_skip_stage("stage2")
            .with_progress(move |name, current, total| {
                progress_calls_clone
                    .lock()
                    .unwrap()
                    .push((name.to_string(), current, total));
            });

        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext::new(&MockBackend, &trace_sink);

        let graph = BuildGraph::default();
        let _result = pipeline.execute(&ctx, graph);

        let calls = progress_calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0], ("stage1".to_string(), 1, 3));
        assert!(calls[1].0.contains("(skipped)"));
        assert_eq!(calls[2], ("stage3".to_string(), 3, 3));
    }

    #[test]
    fn test_validation_stage() {
        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext::new(&MockBackend, &trace_sink);

        let mut graph = BuildGraph::default();
        let result = ValidationStage.execute(&ctx, &mut graph);

        // Should fail because graph is empty
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_stage_with_nodes() {
        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext::new(&MockBackend, &trace_sink);

        let mut graph = BuildGraph::default();
        let config = ModelConfig::new("test", "1.0");
        graph.add_node("test_node", config);

        let result = ValidationStage.execute(&ctx, &mut graph);

        assert!(result.is_ok());
        let events = trace_sink.get_events();
        assert!(events[0].contains("validated 1 nodes"));
    }

    #[test]
    fn test_hashing_stage() {
        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext::new(&MockBackend, &trace_sink);

        let mut graph = BuildGraph::default();
        let config = ModelConfig::new("test", "1.0");
        graph.add_node("test_node", config);

        let result = HashingStage.execute(&ctx, &mut graph);

        assert!(result.is_ok());
        let events = trace_sink.get_events();
        assert!(events[0].contains("computed graph hash:"));
    }

    #[test]
    fn test_multiple_pre_hooks() {
        let hook1_called = Arc::new(Mutex::new(false));
        let hook2_called = Arc::new(Mutex::new(false));

        let hook1_called_clone = Arc::clone(&hook1_called);
        let hook2_called_clone = Arc::clone(&hook2_called);

        let pipeline = BuildPipeline::new()
            .with_pre_hook(move |_ctx| {
                *hook1_called_clone.lock().unwrap() = true;
                Ok(())
            })
            .with_pre_hook(move |_ctx| {
                *hook2_called_clone.lock().unwrap() = true;
                Ok(())
            });

        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext::new(&MockBackend, &trace_sink);

        let graph = BuildGraph::default();
        let _result = pipeline.execute(&ctx, graph);

        assert!(*hook1_called.lock().unwrap());
        assert!(*hook2_called.lock().unwrap());
    }

    #[test]
    fn test_hook_error_propagation() {
        let pipeline =
            BuildPipeline::new().with_pre_hook(|_ctx| Err(AphelionError::validation("test error")));

        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext::new(&MockBackend, &trace_sink);

        let graph = BuildGraph::default();
        let result = pipeline.execute(&ctx, graph);

        assert!(result.is_err());
    }

    #[test]
    fn test_build_context_new() {
        let backend = MockBackend;
        let trace_sink = MockTraceSink::new();

        let ctx = BuildContext::new(&backend, &trace_sink);

        assert_eq!(ctx.backend.name(), "mock");
        assert_eq!(ctx.backend.device(), "mock_device");
        assert!(ctx.backend.is_available());
    }

    #[test]
    fn test_build_context_with_null_backend() {
        let backend = crate::backend::NullBackend::cpu();
        let trace_sink = MockTraceSink::new();

        let ctx = BuildContext::with_null_backend(&backend, &trace_sink);

        assert_eq!(ctx.backend.name(), "null");
        assert_eq!(ctx.backend.device(), "cpu");
        assert!(ctx.backend.is_available());
    }

    #[test]
    fn test_build_context_new_method() {
        let backend = MockBackend;
        let trace_sink = MockTraceSink::new();

        let ctx1 = BuildContext::new(&backend, &trace_sink);
        let ctx2 = BuildContext::new(&backend, &trace_sink);

        assert_eq!(ctx1.backend.name(), ctx2.backend.name());
        assert_eq!(ctx1.backend.device(), ctx2.backend.device());
    }

    #[test]
    fn test_standard_pipeline_has_validation_and_hashing() {
        let pipeline = BuildPipeline::standard();

        // Standard pipeline should have exactly 2 stages
        assert_eq!(pipeline.stages.len(), 2);

        // Verify stage names
        let stage_names: Vec<&str> = pipeline.stages.iter().map(|s| s.name()).collect();
        assert_eq!(stage_names, vec!["validation", "hashing"]);
    }

    #[test]
    fn test_training_pipeline_logs_start() {
        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext::new(&MockBackend, &trace_sink);

        let mut graph = BuildGraph::default();
        let config = ModelConfig::new("test", "1.0");
        graph.add_node("test_node", config);

        let pipeline = BuildPipeline::for_training();

        // Execute the pipeline to trigger the pre-hook
        let _result = pipeline.execute(&ctx, graph);

        let events = trace_sink.get_events();

        // Should have events from pre-hook and both stages
        assert!(events.len() >= 2);

        // Verify the pre-hook logged training start
        assert!(events
            .iter()
            .any(|e| e.contains("pipeline.training") && e.contains("Starting training pipeline")));
    }

    #[test]
    fn test_training_pipeline_has_standard_stages() {
        let pipeline = BuildPipeline::for_training();

        // Training pipeline should have exactly 2 stages (standard) + 1 pre-hook
        assert_eq!(pipeline.stages.len(), 2);

        let stage_names: Vec<&str> = pipeline.stages.iter().map(|s| s.name()).collect();
        assert_eq!(stage_names, vec!["validation", "hashing"]);

        // Verify it has a pre-hook
        assert_eq!(pipeline.hooks.pre_build.len(), 1);
    }

    #[test]
    fn test_inference_pipeline_minimal() {
        let pipeline = BuildPipeline::for_inference();

        // Inference pipeline should have only 1 stage (hashing)
        assert_eq!(pipeline.stages.len(), 1);

        // Verify it only has hashing
        let stage_names: Vec<&str> = pipeline.stages.iter().map(|s| s.name()).collect();
        assert_eq!(stage_names, vec!["hashing"]);

        // Verify no hooks
        assert_eq!(pipeline.hooks.pre_build.len(), 0);
        assert_eq!(pipeline.hooks.post_build.len(), 0);
    }

    #[test]
    fn test_inference_pipeline_execution() {
        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext::new(&MockBackend, &trace_sink);

        let mut graph = BuildGraph::default();
        let config = ModelConfig::new("test", "1.0");
        graph.add_node("test_node", config);

        let pipeline = BuildPipeline::for_inference();

        // Execution should succeed
        let result = pipeline.execute(&ctx, graph);
        assert!(result.is_ok());

        let events = trace_sink.get_events();

        // Should only have hashing stage event
        assert!(events.iter().any(|e| e.contains("computed graph hash")));

        // Should NOT have validation events
        assert!(!events.iter().any(|e| e.contains("validated")));
    }

    #[test]
    fn test_standard_pipeline_execution() {
        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext::new(&MockBackend, &trace_sink);

        let mut graph = BuildGraph::default();
        let config = ModelConfig::new("test", "1.0");
        graph.add_node("test_node", config);

        let pipeline = BuildPipeline::standard();

        // Execution should succeed
        let result = pipeline.execute(&ctx, graph);
        assert!(result.is_ok());

        let events = trace_sink.get_events();

        // Should have both validation and hashing events
        assert!(events.iter().any(|e| e.contains("validated 1 nodes")));
        assert!(events.iter().any(|e| e.contains("computed graph hash")));
    }

    #[test]
    fn test_preset_pipelines_are_extensible() {
        // Verify that preset pipelines can be further customized
        let pipeline = BuildPipeline::standard().with_progress(|name, current, total| {
            let _ = (name, current, total);
        });

        // Should be able to add more stages on top of presets
        let extended = pipeline.with_stage(Box::new(ValidationStage));
        assert_eq!(extended.stages.len(), 3);
    }
}
