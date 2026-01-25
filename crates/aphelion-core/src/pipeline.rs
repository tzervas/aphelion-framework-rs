//! Build pipeline orchestration and stage management.
//!
//! This module provides the infrastructure for defining and executing multi-stage
//! build pipelines with support for hooks, progress tracking, and stage skipping.
//! The pipeline architecture enables composable, reusable build processes.

use crate::backend::{Backend, ModelBuilder};
use crate::config::{ConfigSpec, ModelConfig};
use crate::diagnostics::{TraceEvent, TraceLevel, TraceSink};
use crate::error::{AphelionError, AphelionResult};
use crate::graph::BuildGraph;
use std::collections::HashSet;
use std::time::SystemTime;

/// Execution context containing backend and tracing infrastructure.
///
/// `BuildContext` provides the runtime environment for pipeline stages and builders.
/// It supplies both the computational backend and the tracing sink for recording events.
///
/// # Fields
///
/// * `backend` - The computational backend to use for operations
/// * `trace` - The trace sink for recording diagnostic events
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
/// let ctx = BuildContext {
///     backend: &backend,
///     trace: &trace,
/// };
/// ```
pub struct BuildContext<'a> {
    /// The computational backend
    pub backend: &'a dyn Backend,
    /// The trace sink for recording events
    pub trace: &'a dyn TraceSink,
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
pub struct PipelineHooks {
    /// Callbacks to execute before any pipeline stages
    pub pre_build: Vec<Box<dyn Fn(&BuildContext) -> AphelionResult<()> + Send + Sync>>,
    /// Callbacks to execute after all pipeline stages
    pub post_build: Vec<Box<dyn Fn(&BuildContext, &BuildGraph) -> AphelionResult<()> + Send + Sync>>,
}

impl Default for PipelineHooks {
    fn default() -> Self {
        Self {
            pre_build: Vec::new(),
            post_build: Vec::new(),
        }
    }
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
/// let ctx = BuildContext {
///     backend: &backend,
///     trace: &trace,
/// };
///
/// let result = pipeline.execute(&ctx, graph);
/// assert!(result.is_ok());
/// ```
pub struct BuildPipeline {
    stages: Vec<Box<dyn PipelineStage>>,
    hooks: PipelineHooks,
    skip_stages: HashSet<String>,
    on_progress: Option<Box<dyn Fn(&str, usize, usize) + Send + Sync>>,
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
            hooks: PipelineHooks::default(),
            skip_stages: HashSet::new(),
            on_progress: None,
        }
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
                    progress(&format!("{} (skipped)", stage_name), index + 1, total_stages);
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
    pub fn build_with_validation<M>(
        model: &M,
        ctx: BuildContext<'_>,
    ) -> AphelionResult<BuildGraph>
    where
        M: ModelBuilder<Output = BuildGraph> + ConfigSpec,
    {
        let config = model.config();
        if config.name.trim().is_empty() {
            return Err(AphelionError::InvalidConfig(
                "name cannot be empty".to_string(),
            ));
        }
        if config.version.trim().is_empty() {
            return Err(AphelionError::InvalidConfig(
                "version cannot be empty".to_string(),
            ));
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
/// let ctx = BuildContext {
///     backend: &backend,
///     trace: &trace,
/// };
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
            return Err(AphelionError::Validation(
                "graph must contain at least one node".to_string(),
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
/// let ctx = BuildContext {
///     backend: &backend,
///     trace: &trace,
/// };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
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
        let ctx = BuildContext {
            backend: &MockBackend,
            trace: &trace_sink,
        };

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
        let ctx = BuildContext {
            backend: &MockBackend,
            trace: &trace_sink,
        };

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
        let ctx = BuildContext {
            backend: &MockBackend,
            trace: &trace_sink,
        };

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
        let ctx = BuildContext {
            backend: &MockBackend,
            trace: &trace_sink,
        };

        let graph = BuildGraph::default();
        let _result = pipeline.execute(&ctx, graph);

        let execution_order = executed.lock().unwrap().clone();
        assert_eq!(execution_order, vec!["stage1", "stage3"]);
    }

    #[test]
    fn test_progress_callback() {
        let progress_calls = Arc::new(Mutex::new(Vec::new()));
        let progress_calls_clone = Arc::clone(&progress_calls);

        let stage1 = Box::new(RecordingStage::new("stage1", Arc::new(Mutex::new(Vec::new()))));
        let stage2 = Box::new(RecordingStage::new("stage2", Arc::new(Mutex::new(Vec::new()))));

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
        let ctx = BuildContext {
            backend: &MockBackend,
            trace: &trace_sink,
        };

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

        let stage1 = Box::new(RecordingStage::new("stage1", Arc::new(Mutex::new(Vec::new()))));
        let stage2 = Box::new(RecordingStage::new("stage2", Arc::new(Mutex::new(Vec::new()))));
        let stage3 = Box::new(RecordingStage::new("stage3", Arc::new(Mutex::new(Vec::new()))));

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
        let ctx = BuildContext {
            backend: &MockBackend,
            trace: &trace_sink,
        };

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
        let ctx = BuildContext {
            backend: &MockBackend,
            trace: &trace_sink,
        };

        let mut graph = BuildGraph::default();
        let result = ValidationStage.execute(&ctx, &mut graph);

        // Should fail because graph is empty
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_stage_with_nodes() {
        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext {
            backend: &MockBackend,
            trace: &trace_sink,
        };

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
        let ctx = BuildContext {
            backend: &MockBackend,
            trace: &trace_sink,
        };

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
        let ctx = BuildContext {
            backend: &MockBackend,
            trace: &trace_sink,
        };

        let graph = BuildGraph::default();
        let _result = pipeline.execute(&ctx, graph);

        assert!(*hook1_called.lock().unwrap());
        assert!(*hook2_called.lock().unwrap());
    }

    #[test]
    fn test_hook_error_propagation() {
        let pipeline = BuildPipeline::new().with_pre_hook(|_ctx| {
            Err(AphelionError::Build("test error".to_string()))
        });

        let trace_sink = MockTraceSink::new();
        let ctx = BuildContext {
            backend: &MockBackend,
            trace: &trace_sink,
        };

        let graph = BuildGraph::default();
        let result = pipeline.execute(&ctx, graph);

        assert!(result.is_err());
    }
}
