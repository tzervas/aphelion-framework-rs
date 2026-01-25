#[cfg(test)]
mod tests {
    use aphelion_core::backend::NullBackend;
    use aphelion_core::config::ModelConfig;
    use aphelion_core::diagnostics::{InMemoryTraceSink, TraceLevel, TraceSink};
    use aphelion_core::graph::BuildGraph;
    use aphelion_core::pipeline::{BuildContext, BuildPipeline};
    use aphelion_core::aphelion_model;
    #[cfg(feature = "burn")]
    use aphelion_core::burn_backend::{BurnBackendConfig, BurnDevice};

    #[test]
    fn graph_hash_is_deterministic() {
        let mut graph_a = BuildGraph::default();
        let mut graph_b = BuildGraph::default();

        let config = ModelConfig::new("toy", "0.1.0")
            .with_param("hidden", serde_json::json!(128))
            .with_param("layers", serde_json::json!(4));

        let node_a = graph_a.add_node("toy", config.clone());
        graph_a.add_edge(node_a, node_a);

        let node_b = graph_b.add_node("toy", config);
        graph_b.add_edge(node_b, node_b);

        assert_eq!(graph_a.stable_hash(), graph_b.stable_hash());
    }

    #[aphelion_model]
    struct TestModel {
        config: ModelConfig,
    }

    impl TestModel {
        fn build_graph(
            &self,
            _backend: &dyn aphelion_core::backend::Backend,
            _trace: &dyn aphelion_core::diagnostics::TraceSink,
        ) -> BuildGraph {
            let mut graph = BuildGraph::default();
            let node = graph.add_node("test", self.config.clone());
            graph.add_edge(node, node);
            graph
        }
    }

    #[test]
    fn pipeline_validates_config() {
        let model = TestModel {
            config: ModelConfig::new("test", "0.1.0"),
        };
        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let result = BuildPipeline::build_with_validation(&model, ctx);
        assert!(result.is_ok());
    }

    #[test]
    fn trace_sink_records_events() {
        let trace = InMemoryTraceSink::new();
        trace.record(aphelion_core::diagnostics::TraceEvent {
            id: "test.event".to_string(),
            message: "hello".to_string(),
            timestamp: std::time::SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });
        assert_eq!(trace.events().len(), 1);
    }

    #[test]
    fn macro_build_graph_convention() {
        let model = TestModel {
            config: ModelConfig::new("test", "0.1.0"),
        };
        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };
        let result = BuildPipeline::build_with_validation(&model, ctx);
        assert!(result.is_ok());
    }

    #[cfg(feature = "burn")]
    #[test]
    fn burn_backend_config_defaults() {
        let cfg = BurnBackendConfig::default();
        match cfg.device {
            BurnDevice::Cpu => {}
            _ => panic!("expected CPU default"),
        }
        assert!(!cfg.allow_tf32);
    }
}
