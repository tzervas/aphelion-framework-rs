//! WASM bindings for pipeline execution.

use super::backend::JsNullBackend;
use super::diagnostics::JsInMemoryTraceSink;
use super::graph::JsBuildGraph;
use crate::pipeline::{BuildContext, HashingStage, PipelineStage, ValidationStage};
use wasm_bindgen::prelude::*;

/// Execution context for pipeline runs.
///
/// BuildContext bundles the backend and trace sink needed
/// for pipeline execution.
#[wasm_bindgen]
pub struct JsBuildContext {
    // We need to store owned values since WASM can't handle references well
    backend: JsNullBackend,
    trace: JsInMemoryTraceSink,
}

#[wasm_bindgen]
impl JsBuildContext {
    /// Create a context with a null backend for testing.
    #[wasm_bindgen(js_name = withNullBackend)]
    pub fn with_null_backend() -> Self {
        Self {
            backend: JsNullBackend::cpu(),
            trace: JsInMemoryTraceSink::new(),
        }
    }

    /// Create a context with custom backend and trace sink.
    #[wasm_bindgen(constructor)]
    pub fn new(backend: JsNullBackend, trace: JsInMemoryTraceSink) -> Self {
        Self { backend, trace }
    }

    /// Get the backend name.
    #[wasm_bindgen(js_name = backendName)]
    pub fn backend_name(&self) -> String {
        self.backend.name()
    }

    /// Get the device identifier.
    #[wasm_bindgen(js_name = deviceName)]
    pub fn device_name(&self) -> String {
        self.backend.device()
    }

    /// Get the trace sink.
    #[wasm_bindgen(getter)]
    pub fn trace(&self) -> JsInMemoryTraceSink {
        JsInMemoryTraceSink::new() // Return a new one since we can't clone Arc easily
    }
}

/// Pipeline for executing build stages.
///
/// BuildPipeline executes stages in sequence, with each stage
/// operating on the build graph.
#[wasm_bindgen]
pub struct JsBuildPipeline {
    stages: Vec<String>,
    skip_stages: Vec<String>,
}

#[wasm_bindgen]
impl JsBuildPipeline {
    /// Create an empty pipeline.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            skip_stages: Vec::new(),
        }
    }

    /// Create a standard pipeline with validation and hashing.
    #[wasm_bindgen]
    pub fn standard() -> Self {
        Self {
            stages: vec!["validation".to_string(), "hashing".to_string()],
            skip_stages: Vec::new(),
        }
    }

    /// Create an inference-optimized pipeline.
    #[wasm_bindgen(js_name = forInference)]
    pub fn for_inference() -> Self {
        Self {
            stages: vec!["hashing".to_string()],
            skip_stages: Vec::new(),
        }
    }

    /// Create a training pipeline.
    #[wasm_bindgen(js_name = forTraining)]
    pub fn for_training() -> Self {
        Self {
            stages: vec!["validation".to_string(), "hashing".to_string()],
            skip_stages: Vec::new(),
        }
    }

    /// Add a stage to the pipeline.
    #[wasm_bindgen(js_name = withStage)]
    pub fn with_stage(&mut self, stage_name: &str) -> JsBuildPipeline {
        let mut stages = self.stages.clone();
        stages.push(stage_name.to_string());
        Self {
            stages,
            skip_stages: self.skip_stages.clone(),
        }
    }

    /// Skip a stage during execution.
    #[wasm_bindgen(js_name = withSkipStage)]
    pub fn with_skip_stage(&mut self, stage_name: &str) -> JsBuildPipeline {
        let mut skip_stages = self.skip_stages.clone();
        skip_stages.push(stage_name.to_string());
        Self {
            stages: self.stages.clone(),
            skip_stages,
        }
    }

    /// Execute the pipeline on a graph.
    #[wasm_bindgen]
    pub fn execute(
        &self,
        ctx: &JsBuildContext,
        graph: JsBuildGraph,
    ) -> Result<JsBuildGraph, JsError> {
        let backend = ctx.backend.inner();
        let trace = ctx.trace.inner();
        let rust_ctx = BuildContext::new(backend, trace);

        let mut rust_graph = graph.into_inner();

        // Execute stages based on their names
        for stage_name in &self.stages {
            if self.skip_stages.contains(stage_name) {
                continue;
            }

            match stage_name.as_str() {
                "validation" => {
                    let stage = ValidationStage;
                    stage
                        .execute(&rust_ctx, &mut rust_graph)
                        .map_err(|e| JsError::new(&format!("Validation failed: {}", e)))?;
                }
                "hashing" => {
                    let stage = HashingStage;
                    stage
                        .execute(&rust_ctx, &mut rust_graph)
                        .map_err(|e| JsError::new(&format!("Hashing failed: {}", e)))?;
                }
                _ => {
                    // Unknown stage - skip with warning
                }
            }
        }

        Ok(JsBuildGraph::from_inner(rust_graph))
    }

    /// Get the list of stages.
    #[wasm_bindgen(getter)]
    pub fn stages(&self) -> Vec<String> {
        self.stages.clone()
    }
}

impl Default for JsBuildPipeline {
    fn default() -> Self {
        Self::new()
    }
}
