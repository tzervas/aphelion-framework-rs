//! Python bindings for pipeline execution.
//!
//! Pipelines execute stages in sequence to build and validate model graphs.
//! Each stage performs a specific operation: validation, hashing, optimization, etc.

use pyo3::prelude::*;
use std::sync::Arc;
use std::time::SystemTime;

use crate::backend::NullBackend;
use crate::diagnostics::{InMemoryTraceSink, TraceEvent, TraceLevel};
use crate::error::AphelionError;
use crate::graph::BuildGraph;
use crate::pipeline::{BuildContext, HashingStage, PipelineStage, ValidationStage};

use super::backend::{AnyBackend, PyNullBackend};
use super::diagnostics::{AnyTraceSink, PyInMemoryTraceSink};
use super::graph::PyBuildGraph;

/// Execution context holding backend and trace sink.
///
/// BuildContext bundles the execution environment for a pipeline run:
/// - Backend: hardware target (CPU, GPU, etc.)
/// - Trace sink: destination for diagnostic events
///
/// Why this exists:
/// Decoupling execution context from the pipeline allows the same pipeline
/// definition to run on different hardware or with different logging
/// configurations without modification.
///
/// Example:
///     >>> backend = NullBackend.cpu()
///     >>> trace = InMemoryTraceSink()
///     >>> ctx = BuildContext(backend, trace)
///     >>> # Or use the convenience constructor:
///     >>> ctx = BuildContext.with_null_backend()
#[pyclass(name = "BuildContext")]
pub struct PyBuildContext {
    pub(crate) backend: AnyBackend,
    pub(crate) trace: AnyTraceSink,
}

#[pymethods]
impl PyBuildContext {
    /// Create a build context with explicit backend and trace sink.
    ///
    /// Args:
    ///     backend: Hardware backend for execution.
    ///     trace: Trace sink for diagnostic events.
    #[new]
    #[pyo3(text_signature = "(backend, trace)")]
    fn new(backend: &PyNullBackend, trace: &PyInMemoryTraceSink) -> Self {
        Self {
            backend: AnyBackend::from(backend.clone()),
            trace: AnyTraceSink::from(trace.clone()),
        }
    }

    /// Create a context with NullBackend and in-memory trace sink.
    ///
    /// Convenience constructor for testing and development.
    ///
    /// Returns:
    ///     BuildContext configured for CPU testing.
    ///
    /// Example:
    ///     >>> ctx = BuildContext.with_null_backend()
    #[staticmethod]
    fn with_null_backend() -> Self {
        Self {
            backend: AnyBackend::Null(NullBackend::cpu()),
            trace: AnyTraceSink::InMemory(Arc::new(InMemoryTraceSink::new())),
        }
    }

    fn __repr__(&self) -> String {
        let backend_name = self.backend.as_backend().name();
        format!("BuildContext(backend='{}')", backend_name)
    }
}

/// Pipeline for executing build stages on a graph.
///
/// Pipelines define a sequence of stages that transform or validate a
/// BuildGraph. Common stages include:
/// - validation: Check configuration correctness
/// - hashing: Compute deterministic graph hash
///
/// Why pipelines:
/// - Composability: Mix and match stages for different workflows
/// - Observability: Each stage emits trace events for debugging
/// - Consistency: Standard pipelines ensure consistent build processes
///
/// Example:
///     >>> pipeline = BuildPipeline.standard()
///     >>> result = pipeline.execute(ctx, graph)
///     >>> print(result.stable_hash())
///
/// Preset pipelines:
///     - standard(): Validation + hashing (recommended for most uses)
///     - for_training(): Same as standard, optimized for training workflows
///     - for_inference(): Hashing only, minimal overhead
#[pyclass(name = "BuildPipeline")]
pub struct PyBuildPipeline {
    stages: Vec<String>,
}

/// Execute a named stage on the graph.
fn execute_stage(
    stage_name: &str,
    ctx: &BuildContext<'_>,
    graph: &mut BuildGraph,
) -> Result<(), AphelionError> {
    match stage_name {
        "validation" => ValidationStage.execute(ctx, graph),
        "hashing" => HashingStage.execute(ctx, graph),
        _ => {
            // Unknown stages are logged but don't fail the pipeline
            ctx.trace.record(TraceEvent {
                id: format!("stage.{}", stage_name),
                message: format!("Unknown stage '{}' skipped", stage_name),
                timestamp: SystemTime::now(),
                level: TraceLevel::Warn,
                span_id: None,
                trace_id: None,
            });
            Ok(())
        }
    }
}

#[pymethods]
impl PyBuildPipeline {
    /// Create an empty pipeline with no stages.
    ///
    /// Use with_stage() to add stages, or use a preset like standard().
    #[new]
    fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Standard pipeline with validation and hashing.
    ///
    /// Recommended for most use cases. Validates configuration correctness
    /// before computing the deterministic hash.
    ///
    /// Returns:
    ///     Pipeline configured with [validation, hashing] stages.
    ///
    /// Example:
    ///     >>> pipeline = BuildPipeline.standard()
    #[staticmethod]
    fn standard() -> Self {
        Self {
            stages: vec!["validation".to_string(), "hashing".to_string()],
        }
    }

    /// Pipeline optimized for training workflows.
    ///
    /// Currently identical to standard(). Future versions may add
    /// training-specific stages like gradient checkpointing setup.
    ///
    /// Returns:
    ///     Pipeline configured for training.
    #[staticmethod]
    fn for_training() -> Self {
        Self {
            stages: vec!["validation".to_string(), "hashing".to_string()],
        }
    }

    /// Minimal pipeline for inference.
    ///
    /// Skips validation for reduced overhead. Use only when you're
    /// confident the graph is already valid.
    ///
    /// Returns:
    ///     Pipeline with [hashing] stage only.
    #[staticmethod]
    fn for_inference() -> Self {
        // Inference pipeline skips validation for speed
        Self {
            stages: vec!["hashing".to_string()],
        }
    }

    /// Add a stage to the pipeline.
    ///
    /// Returns a new pipeline with the stage appended.
    ///
    /// Args:
    ///     name: Stage name ("validation", "hashing").
    ///
    /// Returns:
    ///     New pipeline with stage added.
    ///
    /// Example:
    ///     >>> pipeline = BuildPipeline().with_stage("validation")
    #[pyo3(text_signature = "(name)")]
    fn with_stage(&self, name: String) -> Self {
        let mut stages = self.stages.clone();
        stages.push(name);
        Self { stages }
    }

    /// Execute the pipeline synchronously.
    ///
    /// Runs all stages in sequence on the provided graph. Each stage may
    /// modify the graph (e.g., adding computed metadata) or validate it.
    ///
    /// Args:
    ///     ctx: Build context with backend and trace sink.
    ///     graph: Input graph to process.
    ///
    /// Returns:
    ///     Processed graph (may be modified by stages).
    ///
    /// Raises:
    ///     RuntimeError: If any stage fails.
    ///
    /// Example:
    ///     >>> result = pipeline.execute(ctx, graph)
    ///     >>> print(result.stable_hash())
    #[pyo3(text_signature = "(ctx, graph)")]
    fn execute(&self, ctx: &PyBuildContext, graph: PyBuildGraph) -> PyResult<PyBuildGraph> {
        let trace_sink = ctx.trace.as_trace_sink();

        // Record execution start
        trace_sink.record(TraceEvent {
            id: "pipeline.start".to_string(),
            message: format!("Executing pipeline with {} stages", self.stages.len()),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });

        // Create Rust build context
        let rust_ctx = BuildContext::new(ctx.backend.as_backend(), trace_sink);

        // Execute each stage
        let mut rust_graph = graph.inner;
        let total_stages = self.stages.len();

        for (index, stage_name) in self.stages.iter().enumerate() {
            // Record stage start
            trace_sink.record(TraceEvent {
                id: format!("stage.{}.start", stage_name),
                message: format!(
                    "Starting stage '{}' ({}/{})",
                    stage_name,
                    index + 1,
                    total_stages
                ),
                timestamp: SystemTime::now(),
                level: TraceLevel::Debug,
                span_id: None,
                trace_id: None,
            });

            // Execute the stage
            execute_stage(stage_name, &rust_ctx, &mut rust_graph).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Stage '{}' failed: {}",
                    stage_name, e
                ))
            })?;
        }

        // Record execution completion
        trace_sink.record(TraceEvent {
            id: "pipeline.complete".to_string(),
            message: format!(
                "Pipeline completed successfully ({} stages)",
                self.stages.len()
            ),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });

        Ok(PyBuildGraph { inner: rust_graph })
    }

    /// Execute the pipeline asynchronously.
    ///
    /// Async version of execute() for integration with asyncio. Yields
    /// between stages to allow other coroutines to run.
    ///
    /// Args:
    ///     ctx: Build context with backend and trace sink.
    ///     graph: Input graph to process.
    ///
    /// Returns:
    ///     Awaitable that resolves to the processed graph.
    ///
    /// Example:
    ///     >>> result = await pipeline.execute_async(ctx, graph)
    #[pyo3(text_signature = "(ctx, graph)")]
    fn execute_async<'py>(
        &self,
        py: Python<'py>,
        ctx: &PyBuildContext,
        graph: PyBuildGraph,
    ) -> PyResult<Bound<'py, PyAny>> {
        let stages = self.stages.clone();
        let mut rust_graph = graph.inner;
        let backend = ctx.backend.clone();
        let trace = ctx.trace.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Yield to allow other tasks to run
            tokio::task::yield_now().await;

            let trace_sink = trace.as_trace_sink();

            // Record execution start
            trace_sink.record(TraceEvent {
                id: "pipeline.async.start".to_string(),
                message: format!("Executing async pipeline with {} stages", stages.len()),
                timestamp: SystemTime::now(),
                level: TraceLevel::Info,
                span_id: None,
                trace_id: None,
            });

            // Create Rust build context
            let rust_ctx = BuildContext::new(backend.as_backend(), trace_sink);

            // Execute each stage
            for stage_name in &stages {
                // Yield between stages for better async behavior
                tokio::task::yield_now().await;

                execute_stage(stage_name, &rust_ctx, &mut rust_graph).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Stage '{}' failed: {}",
                        stage_name, e
                    ))
                })?;
            }

            // Record execution completion
            trace_sink.record(TraceEvent {
                id: "pipeline.async.complete".to_string(),
                message: format!(
                    "Async pipeline completed successfully ({} stages)",
                    stages.len()
                ),
                timestamp: SystemTime::now(),
                level: TraceLevel::Info,
                span_id: None,
                trace_id: None,
            });

            Ok(PyBuildGraph { inner: rust_graph })
        })
    }

    fn __len__(&self) -> usize {
        self.stages.len()
    }

    fn __repr__(&self) -> String {
        format!("BuildPipeline(stages={:?})", self.stages)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBuildContext>()?;
    m.add_class::<PyBuildPipeline>()?;
    Ok(())
}
