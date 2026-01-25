use aphelion_core::aphelion_model;
use aphelion_core::backend::{Backend, NullBackend};
use aphelion_core::config::ModelConfig;
use aphelion_core::diagnostics::{TraceEvent, TraceSink};
use aphelion_core::graph::BuildGraph;
use aphelion_core::pipeline::{BuildContext, BuildPipeline};
use std::time::SystemTime;

#[aphelion_model]
pub struct ToyModel {
    pub config: ModelConfig,
}

impl ToyModel {
    pub fn new() -> Self {
        Self {
            config: ModelConfig::new("toy_model", "0.1.0")
                .with_param("hidden_size", serde_json::json!(64))
                .with_param("layers", serde_json::json!(2)),
        }
    }
}

impl ToyModel {
    pub fn build_graph(&self, backend: &dyn Backend, trace: &dyn TraceSink) -> BuildGraph {
        trace.record(TraceEvent {
            id: "toy_model.init".to_string(),
            message: format!("building with backend={} device={}", backend.name(), backend.device()),
            timestamp: SystemTime::now(),
        });

        let mut graph = BuildGraph::default();
        let node_id = graph.add_node("toy_model", self.config.clone());
        graph.add_edge(node_id, node_id);

        trace.record(TraceEvent {
            id: "toy_model.graph".to_string(),
            message: format!("graph_hash={}", graph.stable_hash()),
            timestamp: SystemTime::now(),
        });

        graph
    }
}

pub fn build_with_pipeline(model: &ToyModel, backend: &dyn Backend, trace: &dyn TraceSink) {
    let ctx = BuildContext { backend, trace };
    let _ = BuildPipeline::build(model, ctx);
}

pub fn build_with_validation(model: &ToyModel) {
    let backend = NullBackend::cpu();
    let trace = aphelion_core::diagnostics::InMemoryTraceSink::new();
    let ctx = BuildContext {
        backend: &backend,
        trace: &trace,
    };
    let _ = BuildPipeline::build_with_validation(model, ctx);
}
