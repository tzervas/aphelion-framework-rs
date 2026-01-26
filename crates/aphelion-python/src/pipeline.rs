//! Python bindings for pipeline execution.

use pyo3::prelude::*;
use std::sync::Arc;
use std::time::SystemTime;

use aphelion_core::backend::NullBackend;
use aphelion_core::diagnostics::{InMemoryTraceSink, TraceEvent, TraceLevel};
use aphelion_core::error::AphelionError;
use aphelion_core::graph::BuildGraph;
use aphelion_core::pipeline::{BuildContext, HashingStage, PipelineStage, ValidationStage};

use crate::backend::{AnyBackend, PyNullBackend};
use crate::diagnostics::{AnyTraceSink, PyInMemoryTraceSink};
use crate::graph::PyBuildGraph;

/// Execution context for build pipelines.
#[pyclass(name = "BuildContext")]
pub struct PyBuildContext {
    pub(crate) backend: AnyBackend,
    pub(crate) trace: AnyTraceSink,
}

#[pymethods]
impl PyBuildContext {
    #[new]
    fn new(backend: &PyNullBackend, trace: &PyInMemoryTraceSink) -> Self {
        Self {
            backend: AnyBackend::from(backend.clone()),
            trace: AnyTraceSink::from(trace.clone()),
        }
    }

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

/// Simple pipeline that processes a graph.
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
    #[new]
    fn new() -> Self {
        Self { stages: Vec::new() }
    }

    #[staticmethod]
    fn standard() -> Self {
        Self {
            stages: vec!["validation".to_string(), "hashing".to_string()],
        }
    }

    #[staticmethod]
    fn for_training() -> Self {
        Self {
            stages: vec!["validation".to_string(), "hashing".to_string()],
        }
    }

    #[staticmethod]
    fn for_inference() -> Self {
        // Inference pipeline skips validation for speed
        Self {
            stages: vec!["hashing".to_string()],
        }
    }

    fn with_stage(&self, name: String) -> Self {
        let mut stages = self.stages.clone();
        stages.push(name);
        Self { stages }
    }

    /// Execute the pipeline synchronously.
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
