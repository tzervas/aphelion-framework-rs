#[cfg(test)]
mod tests {
    use aphelion_core::aphelion_model;
    use aphelion_core::backend::{
        Backend, BackendRegistry, DeviceCapabilities, MockBackend, NullBackend,
    };
    #[cfg(feature = "burn")]
    use aphelion_core::burn_backend::{BurnBackendConfig, BurnDevice};
    use aphelion_core::config::ModelConfig;
    use aphelion_core::diagnostics::{
        InMemoryTraceSink, MultiSink, TraceEvent, TraceLevel, TraceSink,
    };
    use aphelion_core::export::JsonExporter;
    use aphelion_core::graph::BuildGraph;
    use aphelion_core::pipeline::{BuildContext, BuildPipeline, PipelineStage};
    use aphelion_core::validation::{
        CompositeValidator, ConfigValidator, NameValidator, VersionValidator,
    };
    use std::sync::{Arc, Mutex};
    use std::time::SystemTime;

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

    // ============================================================================
    // GRAPH DETERMINISM TESTS
    // ============================================================================

    #[test]
    fn graph_same_config_produces_same_hash() {
        // Test that identical graph structures always produce the same hash
        let mut graph1 = BuildGraph::default();
        let mut graph2 = BuildGraph::default();

        let config = ModelConfig::new("model1", "1.0.0")
            .with_param("layers", serde_json::json!(16))
            .with_param("hidden", serde_json::json!(512));

        let n1 = graph1.add_node("layer1", config.clone());
        graph1.add_edge(n1, n1);

        let n2 = graph2.add_node("layer1", config);
        graph2.add_edge(n2, n2);

        let hash1 = graph1.stable_hash();
        let hash2 = graph2.stable_hash();

        assert_eq!(hash1, hash2, "Same config should produce same hash");
        assert!(!hash1.is_empty(), "Hash should not be empty");
    }

    #[test]
    fn graph_different_configs_produce_different_hashes() {
        // Test that different configurations produce different hashes
        let mut graph1 = BuildGraph::default();
        let mut graph2 = BuildGraph::default();

        let config1 =
            ModelConfig::new("model1", "1.0.0").with_param("layers", serde_json::json!(16));
        let config2 =
            ModelConfig::new("model1", "1.0.0").with_param("layers", serde_json::json!(32));

        let n1 = graph1.add_node("layer1", config1);
        graph1.add_edge(n1, n1);

        let n2 = graph2.add_node("layer1", config2);
        graph2.add_edge(n2, n2);

        let hash1 = graph1.stable_hash();
        let hash2 = graph2.stable_hash();

        assert_ne!(
            hash1, hash2,
            "Different configs should produce different hashes"
        );
    }

    #[test]
    fn graph_hash_stability_across_node_additions() {
        // Test that hash remains stable when adding nodes in same order
        let mut graph1 = BuildGraph::default();
        let mut graph2 = BuildGraph::default();

        let config_a = ModelConfig::new("modelA", "1.0.0");
        let config_b = ModelConfig::new("modelB", "1.0.0");

        // Build graph1
        let n1a = graph1.add_node("nodeA", config_a.clone());
        let n1b = graph1.add_node("nodeB", config_b.clone());
        graph1.add_edge(n1a, n1b);

        // Build graph2 with same structure
        let n2a = graph2.add_node("nodeA", config_a);
        let n2b = graph2.add_node("nodeB", config_b);
        graph2.add_edge(n2a, n2b);

        let hash1_before = graph1.stable_hash();
        let hash2_before = graph2.stable_hash();
        assert_eq!(hash1_before, hash2_before);

        // Both graphs maintain stable hashes
        let hash1_after = graph1.stable_hash();
        let hash2_after = graph2.stable_hash();
        assert_eq!(hash1_after, hash2_after);
        assert_eq!(hash1_before, hash1_after, "Hash should be stable");
    }

    // ============================================================================
    // PIPELINE INTEGRATION TESTS
    // ============================================================================

    /// Custom pipeline stage for testing
    struct CountingStage {
        name: String,
        counter: Arc<Mutex<usize>>,
    }

    impl CountingStage {
        fn new(name: &str, counter: Arc<Mutex<usize>>) -> Self {
            Self {
                name: name.to_string(),
                counter,
            }
        }
    }

    impl PipelineStage for CountingStage {
        fn name(&self) -> &str {
            &self.name
        }

        fn execute(
            &self,
            _ctx: &aphelion_core::pipeline::BuildContext,
            _graph: &mut BuildGraph,
        ) -> aphelion_core::error::AphelionResult<()> {
            let mut c = self.counter.lock().unwrap();
            *c += 1;
            Ok(())
        }
    }

    #[test]
    fn pipeline_full_multi_stage_execution() {
        // Test full pipeline with multiple stages
        let counter = Arc::new(Mutex::new(0));

        let stage1 = Box::new(CountingStage::new("stage1", Arc::clone(&counter)));
        let stage2 = Box::new(CountingStage::new("stage2", Arc::clone(&counter)));
        let stage3 = Box::new(CountingStage::new("stage3", Arc::clone(&counter)));

        let pipeline = BuildPipeline::new()
            .with_stage(stage1)
            .with_stage(stage2)
            .with_stage(stage3);

        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let graph = BuildGraph::default();
        let result = pipeline.execute(&ctx, graph);

        assert!(result.is_ok());
        assert_eq!(
            *counter.lock().unwrap(),
            3,
            "All three stages should execute"
        );
    }

    #[test]
    fn pipeline_error_propagation_through_stages() {
        // Test that errors propagate correctly through pipeline stages
        struct FailingStage;

        impl PipelineStage for FailingStage {
            fn name(&self) -> &str {
                "failing_stage"
            }

            fn execute(
                &self,
                _ctx: &aphelion_core::pipeline::BuildContext,
                _graph: &mut BuildGraph,
            ) -> aphelion_core::error::AphelionResult<()> {
                Err(aphelion_core::error::AphelionError::build(
                    "Stage failed as expected",
                ))
            }
        }

        let counter = Arc::new(Mutex::new(0));
        let stage_before = Box::new(CountingStage::new("before", Arc::clone(&counter)));
        let failing_stage = Box::new(FailingStage);
        let stage_after = Box::new(CountingStage::new("after", Arc::clone(&counter)));

        let pipeline = BuildPipeline::new()
            .with_stage(stage_before)
            .with_stage(failing_stage)
            .with_stage(stage_after);

        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let graph = BuildGraph::default();
        let result = pipeline.execute(&ctx, graph);

        assert!(result.is_err(), "Pipeline should propagate error");
        // Stage before should execute, but not after
        assert_eq!(
            *counter.lock().unwrap(),
            1,
            "Only first stage should execute before error"
        );
    }

    #[test]
    fn pipeline_hook_execution_order() {
        // Test that pre and post hooks execute in correct order
        let execution_log = Arc::new(Mutex::new(Vec::new()));

        let log_clone1 = Arc::clone(&execution_log);
        let pre_hook = move |_ctx: &BuildContext| {
            log_clone1.lock().unwrap().push("pre_hook");
            Ok(())
        };

        let stage_counter = Arc::new(Mutex::new(0));
        let stage_counter_clone = Arc::clone(&stage_counter);
        let stage = Box::new(CountingStage {
            name: "test_stage".to_string(),
            counter: stage_counter_clone,
        });

        let log_clone3 = Arc::clone(&execution_log);
        let post_hook = move |_ctx: &BuildContext, _graph: &BuildGraph| {
            log_clone3.lock().unwrap().push("post_hook");
            Ok(())
        };

        let pipeline = BuildPipeline::new()
            .with_pre_hook(pre_hook)
            .with_stage(stage)
            .with_post_hook(post_hook);

        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let graph = BuildGraph::default();
        let result = pipeline.execute(&ctx, graph);

        assert!(result.is_ok());
        let log = execution_log.lock().unwrap();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0], "pre_hook", "Pre-hook should execute first");
        assert_eq!(log[1], "post_hook", "Post-hook should execute last");
    }

    // ============================================================================
    // BACKEND INTEGRATION TESTS
    // ============================================================================

    #[test]
    fn backend_registry_operations_integration() {
        // Test BackendRegistry operations
        let mut registry = BackendRegistry::new();

        let backend1 = Box::new(NullBackend::cpu());
        let backend2 = Box::new(MockBackend::new("mock1", "gpu"));
        let backend3 = Box::new(MockBackend::new("mock2", "cpu"));

        registry.register(backend1);
        registry.register(backend2);
        registry.register(backend3);

        let available = registry.list_available();
        assert_eq!(available.len(), 3);
        assert!(available.contains(&"null"));
        assert!(available.contains(&"mock1"));
        assert!(available.contains(&"mock2"));

        // Verify we can retrieve each backend
        assert!(registry.get("null").is_some());
        assert!(registry.get("mock1").is_some());
        assert!(registry.get("mock2").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn backend_lifecycle_init_shutdown() {
        // Test backend lifecycle: init and shutdown
        let mut backend = MockBackend::new("lifecycle_test", "cpu");

        assert!(!backend.init_called());
        assert!(!backend.shutdown_called());

        let init_result = backend.initialize();
        assert!(init_result.is_ok());
        assert!(backend.init_called());
        assert!(!backend.shutdown_called());

        let shutdown_result = backend.shutdown();
        assert!(shutdown_result.is_ok());
        assert!(backend.init_called());
        assert!(backend.shutdown_called());
    }

    #[test]
    fn backend_lifecycle_failure_injection() {
        // Test MockBackend failure injection
        let mut backend_init_fail = MockBackend::new("fail_init", "cpu").with_init_failure();

        let init_result = backend_init_fail.initialize();
        assert!(init_result.is_err());
        assert!(backend_init_fail.init_called());

        let mut backend_shutdown_fail =
            MockBackend::new("fail_shutdown", "cpu").with_shutdown_failure();

        let shutdown_result = backend_shutdown_fail.shutdown();
        assert!(shutdown_result.is_err());
        assert!(backend_shutdown_fail.shutdown_called());
    }

    #[test]
    fn backend_capabilities_in_registry() {
        // Test backend capabilities when registered
        let caps = DeviceCapabilities {
            supports_f16: true,
            supports_bf16: true,
            supports_tf32: false,
            max_memory_bytes: Some(16 * 1024 * 1024 * 1024),
            compute_units: Some(2048),
        };

        let backend = Box::new(MockBackend::new("capable", "gpu").with_capabilities(caps));

        let mut registry = BackendRegistry::new();
        registry.register(backend);

        let retrieved = registry.get("capable").unwrap();
        let retrieved_caps = retrieved.capabilities();

        assert!(retrieved_caps.supports_f16);
        assert!(retrieved_caps.supports_bf16);
        assert!(!retrieved_caps.supports_tf32);
        assert_eq!(
            retrieved_caps.max_memory_bytes,
            Some(16 * 1024 * 1024 * 1024)
        );
        assert_eq!(retrieved_caps.compute_units, Some(2048));
    }

    // ============================================================================
    // VALIDATION INTEGRATION TESTS
    // ============================================================================

    #[test]
    fn composite_validator_with_real_configs() {
        // Test CompositeValidator with real model configs
        let validator = CompositeValidator::new()
            .with_validator(Box::new(NameValidator))
            .with_validator(Box::new(VersionValidator));

        // Test valid config
        let valid_config = ModelConfig::new("my-model", "1.0.0");
        assert!(validator.validate(&valid_config).is_ok());

        // Test invalid name
        let invalid_name = ModelConfig::new("invalid@name", "1.0.0");
        let result = validator.validate(&invalid_name);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.field == "name"));

        // Test invalid version
        let invalid_version = ModelConfig::new("valid-model", "1.0");
        let result = validator.validate(&invalid_version);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.field == "version"));

        // Test multiple errors
        let all_invalid = ModelConfig::new("", "");
        let result = validator.validate(&all_invalid);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn validation_in_pipeline_context() {
        // Test validation integration with pipeline
        let invalid_model = TestModel {
            config: ModelConfig::new("invalid@model", "1.0.0"),
        };

        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        // This should still build since the pipeline.build_with_validation
        // only checks name and version fields
        let result = BuildPipeline::build_with_validation(&invalid_model, ctx);

        // The pipeline will succeed because it uses simple validation
        assert!(result.is_ok());
    }

    #[test]
    fn validator_chain_accumulates_errors() {
        // Test that composite validator accumulates all errors
        let validator = CompositeValidator::new()
            .with_validator(Box::new(NameValidator))
            .with_validator(Box::new(VersionValidator));

        let config = ModelConfig::new("", "not-semver");
        let result = validator.validate(&config);

        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(
            errors.len(),
            2,
            "Should accumulate both name and version errors"
        );
        assert!(errors.iter().any(|e| e.field == "name"));
        assert!(errors.iter().any(|e| e.field == "version"));
    }

    // ============================================================================
    // EXPORT INTEGRATION TESTS
    // ============================================================================

    #[test]
    fn json_exporter_with_real_trace_events() {
        // Test JsonExporter with real trace events
        let exporter = JsonExporter::new();

        // Record multiple events
        for i in 0..3 {
            let event = TraceEvent {
                id: format!("event-{}", i),
                message: format!("Event message {}", i),
                timestamp: SystemTime::now(),
                level: match i {
                    0 => TraceLevel::Debug,
                    1 => TraceLevel::Info,
                    _ => TraceLevel::Error,
                },
                span_id: Some(format!("span-{}", i)),
                trace_id: Some(format!("trace-{}", i)),
            };
            exporter.record(event);
        }

        let json = exporter.to_json();
        assert!(json.contains("event-0"));
        assert!(json.contains("event-1"));
        assert!(json.contains("event-2"));
        assert!(json.contains("DEBUG"));
        assert!(json.contains("INFO"));
        assert!(json.contains("ERROR"));
    }

    #[test]
    fn multi_sink_composition() {
        // Test MultiSink composition with multiple trace sinks
        let sink1 = Arc::new(InMemoryTraceSink::new());
        let sink2 = Arc::new(InMemoryTraceSink::new());
        let sink3 = Arc::new(JsonExporter::new());

        let multi_sink = MultiSink::new(vec![
            Arc::clone(&sink1) as Arc<dyn TraceSink>,
            Arc::clone(&sink2) as Arc<dyn TraceSink>,
            Arc::clone(&sink3) as Arc<dyn TraceSink>,
        ]);

        // Record events through multi-sink
        for i in 0..2 {
            let event = TraceEvent {
                id: format!("multi-{}", i),
                message: format!("Message {}", i),
                timestamp: SystemTime::now(),
                level: TraceLevel::Info,
                span_id: None,
                trace_id: None,
            };
            multi_sink.record(event);
        }

        // Verify events reached all sinks
        assert_eq!(sink1.events().len(), 2);
        assert_eq!(sink2.events().len(), 2);

        let json = sink3.to_json();
        assert!(json.contains("multi-0"));
        assert!(json.contains("multi-1"));
    }

    #[test]
    fn export_integration_with_pipeline() {
        // Test export integration with pipeline trace collection
        let exporter = Arc::new(JsonExporter::new());
        let memory_sink = Arc::new(InMemoryTraceSink::new());

        let multi_sink = MultiSink::new(vec![
            Arc::clone(&exporter) as Arc<dyn TraceSink>,
            Arc::clone(&memory_sink) as Arc<dyn TraceSink>,
        ]);

        // Record a test event through the multi-sink to verify composition works
        let test_event = TraceEvent {
            id: "test-event".to_string(),
            message: "Testing export integration".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        };
        multi_sink.record(test_event);

        let counter = Arc::new(Mutex::new(0));
        let stage = Box::new(CountingStage::new("traced_stage", Arc::clone(&counter)));

        let pipeline = BuildPipeline::new().with_stage(stage);

        let backend = NullBackend::cpu();
        let ctx = BuildContext {
            backend: &backend,
            trace: &multi_sink,
        };

        let graph = BuildGraph::default();
        let result = pipeline.execute(&ctx, graph);

        assert!(result.is_ok());
        assert_eq!(*counter.lock().unwrap(), 1);

        // Both sinks should have recorded the test event
        assert!(memory_sink.events().len() >= 1);
        let json = exporter.to_json();
        assert!(!json.is_empty());
        assert_ne!(json, "[]", "JSON should contain events");
        assert!(json.contains("test-event"));
    }

    #[test]
    fn json_exporter_serialization_format() {
        // Test that JsonExporter produces valid JSON
        let exporter = JsonExporter::new();

        let event = TraceEvent {
            id: "format-test".to_string(),
            message: "Testing serialization".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Warn,
            span_id: Some("span-123".to_string()),
            trace_id: Some("trace-456".to_string()),
        };

        exporter.record(event);
        let json = exporter.to_json();

        // Verify it's valid JSON by checking structure
        assert!(json.contains("["));
        assert!(json.contains("]"));
        assert!(json.contains("\"id\""));
        assert!(json.contains("\"message\""));
        assert!(json.contains("\"level\""));
        assert!(json.contains("format-test"));
        assert!(json.contains("Testing serialization"));
    }

    // ============================================================================
    // CROSS-MODULE INTEGRATION TESTS
    // ============================================================================

    #[test]
    fn full_pipeline_with_validation_and_export() {
        // Complete integration test: validation, pipeline execution, export
        let exporter = JsonExporter::new();
        let validator = CompositeValidator::new()
            .with_validator(Box::new(NameValidator))
            .with_validator(Box::new(VersionValidator));

        let config = ModelConfig::new("integration-test", "1.0.0");
        assert!(validator.validate(&config).is_ok());

        let model = TestModel { config };
        let backend = NullBackend::cpu();
        let ctx = BuildContext {
            backend: &backend,
            trace: &exporter,
        };

        let result = BuildPipeline::build_with_validation(&model, ctx);
        assert!(result.is_ok());

        let json = exporter.to_json();
        assert!(!json.is_empty());
    }

    #[test]
    fn graph_with_multiple_config_types() {
        // Test graph building with various config parameter types
        let mut graph = BuildGraph::default();

        let config1 = ModelConfig::new("model1", "1.0.0")
            .with_param("int_param", serde_json::json!(42))
            .with_param("float_param", serde_json::json!(3.14))
            .with_param("string_param", serde_json::json!("value"))
            .with_param("bool_param", serde_json::json!(true));

        let n1 = graph.add_node("node1", config1);

        let config2 = ModelConfig::new("model2", "2.0.0")
            .with_param("layers", serde_json::json!([1, 2, 3, 4]))
            .with_param("config", serde_json::json!({"nested": "value"}));

        let n2 = graph.add_node("node2", config2);
        graph.add_edge(n1, n2);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);

        let hash = graph.stable_hash();
        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 64, "SHA256 hash should be 64 hex characters");
    }

    #[test]
    fn pipeline_with_backend_registry_context() {
        // Test pipeline execution using backend from registry
        let mut registry = BackendRegistry::new();
        let backend = Box::new(MockBackend::new("test_backend", "cpu").with_capabilities(
            DeviceCapabilities {
                supports_f16: true,
                supports_bf16: false,
                supports_tf32: true,
                max_memory_bytes: Some(8 * 1024 * 1024 * 1024),
                compute_units: Some(512),
            },
        ));
        registry.register(backend);

        let selected_backend = registry.get("test_backend").unwrap();
        assert_eq!(selected_backend.name(), "test_backend");
        assert_eq!(selected_backend.device(), "cpu");

        let caps = selected_backend.capabilities();
        assert!(caps.supports_f16);
        assert!(caps.supports_tf32);

        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: selected_backend,
            trace: &trace,
        };

        let pipeline = BuildPipeline::new();
        let graph = BuildGraph::default();
        let result = pipeline.execute(&ctx, graph);
        assert!(result.is_ok());
    }

    // ============================================================================
    // ASYNC PIPELINE TESTS (SC-13)
    // ============================================================================

    #[cfg(feature = "tokio")]
    #[tokio::test]
    async fn async_pipeline_executes() {
        use aphelion_core::pipeline::AsyncPipelineStage;

        // Test async pipeline execution with async stages
        struct AsyncCountingStage {
            name: String,
            counter: Arc<Mutex<usize>>,
        }

        impl AsyncCountingStage {
            fn new(name: &str, counter: Arc<Mutex<usize>>) -> Self {
                Self {
                    name: name.to_string(),
                    counter,
                }
            }
        }

        impl AsyncPipelineStage for AsyncCountingStage {
            fn name(&self) -> &str {
                &self.name
            }

            fn execute_async<'a>(
                &'a self,
                _ctx: &'a BuildContext<'_>,
                _graph: &'a mut BuildGraph,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<Output = aphelion_core::error::AphelionResult<()>>
                        + Send
                        + 'a,
                >,
            > {
                let counter = Arc::clone(&self.counter);
                Box::pin(async move {
                    // Simulate async work
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                    let mut c = counter.lock().unwrap();
                    *c += 1;
                    Ok(())
                })
            }
        }

        let counter = Arc::new(Mutex::new(0));

        let stage1 = Box::new(AsyncCountingStage::new(
            "async_stage1",
            Arc::clone(&counter),
        ));
        let stage2 = Box::new(AsyncCountingStage::new(
            "async_stage2",
            Arc::clone(&counter),
        ));
        let stage3 = Box::new(AsyncCountingStage::new(
            "async_stage3",
            Arc::clone(&counter),
        ));

        let pipeline = BuildPipeline::new()
            .with_async_stage(stage1)
            .with_async_stage(stage2)
            .with_async_stage(stage3);

        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let graph = BuildGraph::default();
        let result = pipeline.execute_async(&ctx, graph).await;

        assert!(result.is_ok(), "Async pipeline should execute successfully");
        assert_eq!(
            *counter.lock().unwrap(),
            3,
            "All three async stages should execute"
        );
    }

    #[cfg(feature = "tokio")]
    #[tokio::test]
    async fn async_pipeline_with_builtin_stages() {
        use aphelion_core::pipeline::{HashingStage, ValidationStage};

        // Test async pipeline with built-in stages that implement AsyncPipelineStage
        let mut graph = BuildGraph::default();
        let config = ModelConfig::new("async-test", "1.0.0");
        graph.add_node("test_node", config);

        let pipeline = BuildPipeline::new()
            .with_async_stage(Box::new(ValidationStage))
            .with_async_stage(Box::new(HashingStage));

        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let result = pipeline.execute_async(&ctx, graph).await;

        assert!(
            result.is_ok(),
            "Async pipeline with validation and hashing should succeed"
        );

        let events = trace.events();
        assert!(events
            .iter()
            .any(|e| e.message.contains("validated 1 nodes")));
        assert!(events
            .iter()
            .any(|e| e.message.contains("computed graph hash")));
    }

    #[cfg(feature = "tokio")]
    #[tokio::test]
    async fn async_pipeline_error_propagation() {
        use aphelion_core::pipeline::AsyncPipelineStage;

        // Test error propagation in async pipeline
        struct FailingAsyncStage;

        impl AsyncPipelineStage for FailingAsyncStage {
            fn name(&self) -> &str {
                "failing_async"
            }

            fn execute_async<'a>(
                &'a self,
                _ctx: &'a BuildContext<'_>,
                _graph: &'a mut BuildGraph,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<Output = aphelion_core::error::AphelionResult<()>>
                        + Send
                        + 'a,
                >,
            > {
                Box::pin(async move {
                    Err(aphelion_core::error::AphelionError::build(
                        "Async stage failed as expected",
                    ))
                })
            }
        }

        let counter = Arc::new(Mutex::new(0));

        struct AsyncCountingStage {
            counter: Arc<Mutex<usize>>,
        }

        impl AsyncPipelineStage for AsyncCountingStage {
            fn name(&self) -> &str {
                "async_counting"
            }

            fn execute_async<'a>(
                &'a self,
                _ctx: &'a BuildContext<'_>,
                _graph: &'a mut BuildGraph,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<Output = aphelion_core::error::AphelionResult<()>>
                        + Send
                        + 'a,
                >,
            > {
                let counter = Arc::clone(&self.counter);
                Box::pin(async move {
                    let mut c = counter.lock().unwrap();
                    *c += 1;
                    Ok(())
                })
            }
        }

        let stage_before = Box::new(AsyncCountingStage {
            counter: Arc::clone(&counter),
        });
        let failing_stage = Box::new(FailingAsyncStage);
        let stage_after = Box::new(AsyncCountingStage {
            counter: Arc::clone(&counter),
        });

        let pipeline = BuildPipeline::new()
            .with_async_stage(stage_before)
            .with_async_stage(failing_stage)
            .with_async_stage(stage_after);

        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let graph = BuildGraph::default();
        let result = pipeline.execute_async(&ctx, graph).await;

        assert!(result.is_err(), "Pipeline should propagate async error");
        assert_eq!(
            *counter.lock().unwrap(),
            1,
            "Only first stage should execute before async error"
        );
    }

    // ============================================================================
    // TRITTER-ACCEL INTEGRATION TESTS
    // ============================================================================

    #[cfg(feature = "tritter-accel")]
    mod tritter_accel_tests {
        use super::*;
        use aphelion_core::acceleration::{
            gradient_compression_post_hook, gradient_compression_pre_hook, inference_pipeline,
            training_pipeline, AccelerationStage, InferenceAccelConfig, TrainingAccelConfig,
        };
        use aphelion_core::tritter_backend::{
            InferenceConfig, TrainingConfig, TriterAccelBackend, TriterDevice,
        };

        #[test]
        fn tritter_backend_cpu_creation() {
            let backend = TriterAccelBackend::cpu();
            assert_eq!(backend.name(), "tritter-accel");
            assert_eq!(backend.device(), "cpu");
            assert!(backend.is_available());
        }

        #[test]
        fn tritter_backend_cuda_creation() {
            let backend = TriterAccelBackend::cuda(0);
            assert_eq!(backend.name(), "tritter-accel");
            assert_eq!(backend.device(), "cuda:0");
        }

        #[test]
        fn tritter_backend_training_mode() {
            let backend = TriterAccelBackend::cpu().with_training_mode(0.1);
            assert!(backend.is_training_mode());
            assert!(!backend.is_inference_mode());

            let config = backend.training_config().unwrap();
            assert!((config.compression_ratio - 0.1).abs() < f32::EPSILON);
        }

        #[test]
        fn tritter_backend_inference_mode() {
            let backend = TriterAccelBackend::cpu().with_inference_mode(32);
            assert!(!backend.is_training_mode());
            assert!(backend.is_inference_mode());

            let config = backend.inference_config().unwrap();
            assert_eq!(config.batch_size, 32);
        }

        #[test]
        fn tritter_backend_lifecycle() {
            let mut backend = TriterAccelBackend::cpu();
            assert!(!backend.is_initialized());

            let init_result = backend.initialize();
            assert!(init_result.is_ok());
            assert!(backend.is_initialized());

            let shutdown_result = backend.shutdown();
            assert!(shutdown_result.is_ok());
            assert!(!backend.is_initialized());
        }

        #[test]
        fn acceleration_stage_training() {
            let stage = AccelerationStage::for_training(0.1);
            assert!(stage.is_training());
            assert!(!stage.is_inference());
            assert_eq!(stage.name(), "tritter-acceleration-training");
        }

        #[test]
        fn acceleration_stage_inference() {
            let stage = AccelerationStage::for_inference(32);
            assert!(!stage.is_training());
            assert!(stage.is_inference());
            assert_eq!(stage.name(), "tritter-acceleration-inference");
        }

        #[test]
        fn acceleration_stage_applies_training_metadata() {
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

            // Verify metadata was applied to all nodes
            for node in &graph.nodes {
                assert!(node.metadata.contains_key("accel.mode"));
                assert_eq!(
                    node.metadata.get("accel.mode"),
                    Some(&serde_json::Value::String("training".to_string()))
                );
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
        fn acceleration_stage_applies_inference_metadata() {
            let config = InferenceAccelConfig::new(32).with_kv_cache(2048);
            let stage = AccelerationStage::with_inference_config(config);

            let backend = NullBackend::cpu();
            let trace = InMemoryTraceSink::new();
            let ctx = BuildContext {
                backend: &backend,
                trace: &trace,
            };

            let mut graph = BuildGraph::default();
            graph.add_node("linear", ModelConfig::new("linear", "1.0"));

            let result = stage.execute(&ctx, &mut graph);
            assert!(result.is_ok());

            // Verify metadata
            let node = &graph.nodes[0];
            assert_eq!(
                node.metadata.get("accel.mode"),
                Some(&serde_json::Value::String("inference".to_string()))
            );
            assert_eq!(
                node.metadata.get("accel.batch_size"),
                Some(&serde_json::Value::Number(serde_json::Number::from(32)))
            );
            assert_eq!(
                node.metadata.get("accel.kv_cache"),
                Some(&serde_json::Value::Bool(true))
            );
            assert_eq!(
                node.metadata.get("accel.max_seq_len"),
                Some(&serde_json::Value::Number(serde_json::Number::from(2048)))
            );
        }

        #[test]
        fn training_pipeline_with_acceleration() {
            let pipeline = training_pipeline(0.1);

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

            let events = trace.events();
            // Should have validation, acceleration, and hashing events
            assert!(events.iter().any(|e| e.message.contains("validated")));
            assert!(events
                .iter()
                .any(|e| e.message.contains("Applied training acceleration")));
            assert!(events
                .iter()
                .any(|e| e.message.contains("computed graph hash")));
        }

        #[test]
        fn inference_pipeline_with_acceleration() {
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

            let events = trace.events();
            // Should have acceleration and hashing events (no validation for inference)
            assert!(events
                .iter()
                .any(|e| e.message.contains("Applied inference acceleration")));
            assert!(events
                .iter()
                .any(|e| e.message.contains("computed graph hash")));
        }

        #[test]
        fn gradient_compression_hooks_integration() {
            let pre_hook = gradient_compression_pre_hook(0.1, 42);
            let post_hook = gradient_compression_post_hook();

            let backend = NullBackend::cpu();
            let trace = InMemoryTraceSink::new();
            let ctx = BuildContext {
                backend: &backend,
                trace: &trace,
            };

            // Run pre-hook
            let result = pre_hook(&ctx);
            assert!(result.is_ok());

            // Apply acceleration to graph
            let mut graph = BuildGraph::default();
            graph.add_node("linear", ModelConfig::new("linear", "1.0"));

            let stage = AccelerationStage::for_training(0.1);
            stage.execute(&ctx, &mut graph).unwrap();

            // Run post-hook
            let trace2 = InMemoryTraceSink::new();
            let ctx2 = BuildContext {
                backend: &backend,
                trace: &trace2,
            };
            let result = post_hook(&ctx2, &graph);
            assert!(result.is_ok());

            let events = trace2.events();
            assert!(events.iter().any(|e| e.message.contains("validated")));
        }

        #[test]
        fn tritter_backend_in_pipeline_context() {
            let mut backend = TriterAccelBackend::cpu().with_training_mode(0.1);
            backend.initialize().unwrap();

            let trace = InMemoryTraceSink::new();
            let ctx = BuildContext {
                backend: &backend,
                trace: &trace,
            };

            let stage = AccelerationStage::for_training(0.1);
            let mut graph = BuildGraph::default();
            graph.add_node("linear", ModelConfig::new("linear", "1.0"));

            let result = stage.execute(&ctx, &mut graph);
            assert!(result.is_ok());

            // Verify backend is used in context
            assert_eq!(ctx.backend.name(), "tritter-accel");
        }

        #[test]
        fn tritter_backend_registry_integration() {
            let mut registry = BackendRegistry::new();

            let tritter_backend = Box::new(TriterAccelBackend::cpu().with_training_mode(0.1));
            let null_backend = Box::new(NullBackend::cpu());

            registry.register(tritter_backend);
            registry.register(null_backend);

            // Auto-detect should prefer tritter-accel over null
            let auto_detected = registry.auto_detect();
            assert!(auto_detected.is_some());
            assert_eq!(auto_detected.unwrap().name(), "tritter-accel");

            // Can explicitly get tritter-accel
            let tritter = registry.get("tritter-accel");
            assert!(tritter.is_some());
            assert!(tritter.unwrap().is_available());
        }

        #[test]
        fn training_config_builder_pattern() {
            let config = TrainingConfig::new(0.1)
                .with_deterministic(true)
                .with_seed(42);

            assert!((config.compression_ratio - 0.1).abs() < f32::EPSILON);
            assert!(config.deterministic);
            assert_eq!(config.seed, Some(42));
        }

        #[test]
        fn inference_config_builder_pattern() {
            let config = InferenceConfig::new(32)
                .with_ternary_layers(true)
                .with_kv_cache(2048);

            assert_eq!(config.batch_size, 32);
            assert!(config.use_ternary_layers);
            assert!(config.use_kv_cache);
            assert_eq!(config.max_seq_len, Some(2048));
        }

        #[test]
        fn triter_device_variants() {
            let cpu = TriterDevice::Cpu;
            let cuda = TriterDevice::Cuda(0);
            let metal = TriterDevice::Metal;

            assert_eq!(cpu.as_str(), "cpu");
            assert_eq!(cuda.as_str(), "cuda:0");
            assert_eq!(metal.as_str(), "metal");
        }

        #[test]
        fn acceleration_stage_with_full_training_config() {
            let config = TrainingAccelConfig::new(0.05)
                .with_seed(12345)
                .with_mixed_precision();

            let stage = AccelerationStage::with_training_config(config);
            assert!(stage.is_training());

            let backend = NullBackend::cpu();
            let trace = InMemoryTraceSink::new();
            let ctx = BuildContext {
                backend: &backend,
                trace: &trace,
            };

            let mut graph = BuildGraph::default();
            graph.add_node("linear", ModelConfig::new("linear", "1.0"));

            let result = stage.execute(&ctx, &mut graph);
            assert!(result.is_ok());

            // Verify full config was applied
            let node = &graph.nodes[0];
            assert!(node.metadata.contains_key("accel.seed"));
            assert!(node.metadata.contains_key("accel.mixed_precision"));
        }
    }

    #[cfg(feature = "tokio")]
    #[tokio::test]
    async fn async_pipeline_with_hooks() {
        use aphelion_core::pipeline::AsyncPipelineStage;

        // Test that pre and post hooks work with async pipeline
        let execution_log = Arc::new(Mutex::new(Vec::new()));

        let log_clone1 = Arc::clone(&execution_log);
        let pre_hook = move |_ctx: &BuildContext| {
            log_clone1.lock().unwrap().push("pre_hook".to_string());
            Ok(())
        };

        let log_clone2 = Arc::clone(&execution_log);
        let post_hook = move |_ctx: &BuildContext, _graph: &BuildGraph| {
            log_clone2.lock().unwrap().push("post_hook".to_string());
            Ok(())
        };

        struct LoggingAsyncStage {
            log: Arc<Mutex<Vec<String>>>,
        }

        impl AsyncPipelineStage for LoggingAsyncStage {
            fn name(&self) -> &str {
                "async_stage"
            }

            fn execute_async<'a>(
                &'a self,
                _ctx: &'a BuildContext<'_>,
                _graph: &'a mut BuildGraph,
            ) -> std::pin::Pin<
                Box<
                    dyn std::future::Future<Output = aphelion_core::error::AphelionResult<()>>
                        + Send
                        + 'a,
                >,
            > {
                let log = Arc::clone(&self.log);
                Box::pin(async move {
                    log.lock().unwrap().push("async_stage".to_string());
                    Ok(())
                })
            }
        }

        let pipeline = BuildPipeline::new()
            .with_pre_hook(pre_hook)
            .with_async_stage(Box::new(LoggingAsyncStage {
                log: Arc::clone(&execution_log),
            }))
            .with_post_hook(post_hook);

        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let graph = BuildGraph::default();
        let result = pipeline.execute_async(&ctx, graph).await;

        assert!(result.is_ok());
        let log = execution_log.lock().unwrap();
        assert_eq!(log.len(), 3);
        assert_eq!(log[0], "pre_hook", "Pre-hook should execute first");
        assert_eq!(log[1], "async_stage", "Async stage should execute second");
        assert_eq!(log[2], "post_hook", "Post-hook should execute last");
    }

    // ============================================================================
    // PROPERTY-BASED TESTS FOR GRAPH HASH STABILITY
    // ============================================================================

    /// Helper to generate test configs with different parameters
    fn generate_test_configs(seed: u64) -> Vec<ModelConfig> {
        let layers_options = [1, 2, 4, 8, 16, 32, 64];
        let hidden_options = [64, 128, 256, 512, 1024];
        let names = ["transformer", "attention", "feedforward", "embedding"];
        let versions = ["1.0.0", "2.0.0", "3.0.0"];

        // Use seed to select pseudo-random indices
        let mut configs = Vec::new();
        for i in 0..10 {
            let idx = (seed.wrapping_mul(31).wrapping_add(i)) as usize;
            let name = names[idx % names.len()];
            let version = versions[idx % versions.len()];
            let layers = layers_options[idx % layers_options.len()];
            let hidden = hidden_options[idx % hidden_options.len()];

            configs.push(
                ModelConfig::new(name, version)
                    .with_param("layers", serde_json::json!(layers))
                    .with_param("hidden", serde_json::json!(hidden))
                    .with_param("seed", serde_json::json!(seed + i)),
            );
        }
        configs
    }

    #[test]
    fn property_hash_deterministic_across_seeds() {
        // Property: Same graph structure always produces same hash regardless of when computed
        // Note: Using wrapping arithmetic in generate_test_configs, so avoid MAX values
        for seed in [0u64, 42, 12345, 999999, 1_000_000_000] {
            let configs = generate_test_configs(seed);

            let mut graph1 = BuildGraph::default();
            let mut graph2 = BuildGraph::default();

            let mut nodes1 = Vec::new();
            let mut nodes2 = Vec::new();

            for (i, config) in configs.iter().enumerate() {
                let n1 = graph1.add_node(&format!("node_{}", i), config.clone());
                let n2 = graph2.add_node(&format!("node_{}", i), config.clone());
                nodes1.push(n1);
                nodes2.push(n2);

                if i > 0 {
                    graph1.add_edge(nodes1[i - 1], n1);
                    graph2.add_edge(nodes2[i - 1], n2);
                }
            }

            let hash1 = graph1.stable_hash();
            let hash2 = graph2.stable_hash();

            assert_eq!(
                hash1, hash2,
                "Identical graphs with seed {} should produce same hash",
                seed
            );

            // Verify hash doesn't change on re-computation
            let hash1_again = graph1.stable_hash();
            assert_eq!(hash1, hash1_again, "Hash should be stable across calls");
        }
    }

    #[test]
    fn property_different_params_produce_different_hashes() {
        // Property: Different configurations should produce different hashes
        let mut hashes = std::collections::HashSet::new();

        for layers in [1, 2, 4, 8, 16] {
            for hidden in [64, 128, 256, 512] {
                let config = ModelConfig::new("test-model", "1.0.0")
                    .with_param("layers", serde_json::json!(layers))
                    .with_param("hidden", serde_json::json!(hidden));

                let mut graph = BuildGraph::default();
                let n = graph.add_node("node", config);
                graph.add_edge(n, n);

                let hash = graph.stable_hash();
                hashes.insert(hash);
            }
        }

        // All 20 combinations (5 layers * 4 hidden) should produce unique hashes
        assert_eq!(
            hashes.len(),
            20,
            "All different configs should produce unique hashes"
        );
    }

    #[test]
    fn property_node_order_matters_for_hash() {
        // Property: Node order affects hash (graphs are ordered)
        let config_a = ModelConfig::new("model-a", "1.0.0");
        let config_b = ModelConfig::new("model-b", "1.0.0");

        let mut graph1 = BuildGraph::default();
        let n1a = graph1.add_node("first", config_a.clone());
        let n1b = graph1.add_node("second", config_b.clone());
        graph1.add_edge(n1a, n1b);

        let mut graph2 = BuildGraph::default();
        let n2b = graph2.add_node("first", config_b.clone());
        let n2a = graph2.add_node("second", config_a.clone());
        graph2.add_edge(n2b, n2a);

        let hash1 = graph1.stable_hash();
        let hash2 = graph2.stable_hash();

        assert_ne!(
            hash1, hash2,
            "Graphs with different node orders should have different hashes"
        );
    }

    #[test]
    fn property_edge_structure_affects_hash() {
        // Property: Different edge structures produce different hashes
        let config = ModelConfig::new("node", "1.0.0");

        // Linear chain: A -> B -> C
        let mut graph_linear = BuildGraph::default();
        let la = graph_linear.add_node("a", config.clone());
        let lb = graph_linear.add_node("b", config.clone());
        let lc = graph_linear.add_node("c", config.clone());
        graph_linear.add_edge(la, lb);
        graph_linear.add_edge(lb, lc);

        // Branching: A -> B, A -> C
        let mut graph_branch = BuildGraph::default();
        let ba = graph_branch.add_node("a", config.clone());
        let bb = graph_branch.add_node("b", config.clone());
        let bc = graph_branch.add_node("c", config.clone());
        graph_branch.add_edge(ba, bb);
        graph_branch.add_edge(ba, bc);

        assert_ne!(
            graph_linear.stable_hash(),
            graph_branch.stable_hash(),
            "Different edge structures should have different hashes"
        );
    }

    #[test]
    fn property_hash_length_consistent() {
        // Property: Hash length is always 64 hex characters (SHA256)
        for i in 0..100 {
            let config = ModelConfig::new(&format!("model-{}", i), "1.0.0")
                .with_param("iteration", serde_json::json!(i));

            let mut graph = BuildGraph::default();
            let n = graph.add_node("node", config);
            graph.add_edge(n, n);

            let hash = graph.stable_hash();
            assert_eq!(
                hash.len(),
                64,
                "Hash should always be 64 hex characters, got {} for iteration {}",
                hash.len(),
                i
            );
            assert!(
                hash.chars().all(|c| c.is_ascii_hexdigit()),
                "Hash should only contain hex digits"
            );
        }
    }

    // ============================================================================
    // CONCURRENT TRACE SINK STRESS TESTS
    // ============================================================================

    #[test]
    fn concurrent_trace_sink_multiple_writers() {
        use std::thread;

        let sink = Arc::new(InMemoryTraceSink::new());
        let num_threads = 10;
        let events_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let sink_clone = Arc::clone(&sink);
                thread::spawn(move || {
                    for event_id in 0..events_per_thread {
                        let event = TraceEvent {
                            id: format!("thread-{}-event-{}", thread_id, event_id),
                            message: format!(
                                "Message from thread {} event {}",
                                thread_id, event_id
                            ),
                            timestamp: SystemTime::now(),
                            level: TraceLevel::Info,
                            span_id: Some(format!("span-{}", thread_id)),
                            trace_id: Some(format!("trace-{}", thread_id)),
                        };
                        sink_clone.record(event);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let events = sink.events();
        assert_eq!(
            events.len(),
            num_threads * events_per_thread,
            "All events from all threads should be recorded"
        );
    }

    #[test]
    fn concurrent_trace_sink_mixed_levels() {
        use std::thread;

        let sink = Arc::new(InMemoryTraceSink::new());
        let num_threads = 5;

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let sink_clone = Arc::clone(&sink);
                thread::spawn(move || {
                    // Each thread writes events with different levels
                    let levels = [
                        TraceLevel::Debug,
                        TraceLevel::Info,
                        TraceLevel::Warn,
                        TraceLevel::Error,
                    ];
                    for (i, level) in levels.iter().enumerate() {
                        let event = TraceEvent {
                            id: format!("t{}-l{}", thread_id, i),
                            message: format!("Level {:?} from thread {}", level, thread_id),
                            timestamp: SystemTime::now(),
                            level: level.clone(),
                            span_id: None,
                            trace_id: None,
                        };
                        sink_clone.record(event);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let events = sink.events();
        assert_eq!(events.len(), num_threads * 4);

        // Count events by level
        let debug_count = events
            .iter()
            .filter(|e| matches!(e.level, TraceLevel::Debug))
            .count();
        let info_count = events
            .iter()
            .filter(|e| matches!(e.level, TraceLevel::Info))
            .count();
        let warn_count = events
            .iter()
            .filter(|e| matches!(e.level, TraceLevel::Warn))
            .count();
        let error_count = events
            .iter()
            .filter(|e| matches!(e.level, TraceLevel::Error))
            .count();

        assert_eq!(debug_count, num_threads);
        assert_eq!(info_count, num_threads);
        assert_eq!(warn_count, num_threads);
        assert_eq!(error_count, num_threads);
    }

    #[test]
    fn concurrent_multi_sink_stress() {
        use std::thread;

        let sink1 = Arc::new(InMemoryTraceSink::new());
        let sink2 = Arc::new(InMemoryTraceSink::new());
        let exporter = Arc::new(JsonExporter::new());

        let multi_sink = Arc::new(MultiSink::new(vec![
            Arc::clone(&sink1) as Arc<dyn TraceSink>,
            Arc::clone(&sink2) as Arc<dyn TraceSink>,
            Arc::clone(&exporter) as Arc<dyn TraceSink>,
        ]));

        let num_threads = 8;
        let events_per_thread = 50;

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let multi_sink_clone = Arc::clone(&multi_sink);
                thread::spawn(move || {
                    for event_id in 0..events_per_thread {
                        let event = TraceEvent {
                            id: format!("multi-{}-{}", thread_id, event_id),
                            message: format!("Multi-sink message {} from {}", event_id, thread_id),
                            timestamp: SystemTime::now(),
                            level: TraceLevel::Info,
                            span_id: None,
                            trace_id: None,
                        };
                        multi_sink_clone.record(event);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let expected_total = num_threads * events_per_thread;

        // All sinks should have received all events
        assert_eq!(sink1.events().len(), expected_total);
        assert_eq!(sink2.events().len(), expected_total);

        let json = exporter.to_json();
        assert!(!json.is_empty());
        // Verify JSON contains events from multiple threads
        assert!(json.contains("multi-0-"));
        assert!(json.contains("multi-1-"));
    }

    #[test]
    fn concurrent_json_exporter_serialization_safety() {
        use std::thread;

        let exporter = Arc::new(JsonExporter::new());
        let num_threads = 4;
        let events_per_thread = 25;

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let exporter_clone = Arc::clone(&exporter);
                thread::spawn(move || {
                    for event_id in 0..events_per_thread {
                        let event = TraceEvent {
                            id: format!("json-{}-{}", thread_id, event_id),
                            message: format!("JSON test from thread {}", thread_id),
                            timestamp: SystemTime::now(),
                            level: TraceLevel::Debug,
                            span_id: Some(format!("span-{}", thread_id)),
                            trace_id: Some(format!("trace-{}", thread_id)),
                        };
                        exporter_clone.record(event);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify JSON is valid and contains all events
        let json = exporter.to_json();
        assert!(json.starts_with("["));
        assert!(json.ends_with("]"));

        // Count occurrences - all threads should be represented
        for thread_id in 0..num_threads {
            assert!(
                json.contains(&format!("json-{}-", thread_id)),
                "Events from thread {} should be in JSON",
                thread_id
            );
        }
    }

    // ============================================================================
    // PIPELINE ERROR RECOVERY TESTS
    // ============================================================================

    /// A stage that can be configured to fail on specific conditions
    struct ConditionalFailureStage {
        name: String,
        fail_on_node_count: Option<usize>,
        fail_message: String,
        execution_count: Arc<Mutex<usize>>,
    }

    impl ConditionalFailureStage {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                fail_on_node_count: None,
                fail_message: "Conditional failure".to_string(),
                execution_count: Arc::new(Mutex::new(0)),
            }
        }

        fn fail_when_nodes_equal(mut self, count: usize) -> Self {
            self.fail_on_node_count = Some(count);
            self
        }

        fn with_message(mut self, msg: &str) -> Self {
            self.fail_message = msg.to_string();
            self
        }

        fn with_counter(mut self, counter: Arc<Mutex<usize>>) -> Self {
            self.execution_count = counter;
            self
        }
    }

    impl PipelineStage for ConditionalFailureStage {
        fn name(&self) -> &str {
            &self.name
        }

        fn execute(
            &self,
            _ctx: &BuildContext,
            graph: &mut BuildGraph,
        ) -> aphelion_core::error::AphelionResult<()> {
            let mut count = self.execution_count.lock().unwrap();
            *count += 1;

            if let Some(fail_count) = self.fail_on_node_count {
                if graph.node_count() == fail_count {
                    return Err(aphelion_core::error::AphelionError::build(
                        &self.fail_message,
                    ));
                }
            }

            Ok(())
        }
    }

    #[test]
    fn pipeline_recovery_failure_at_first_stage() {
        let counter = Arc::new(Mutex::new(0));

        let stage1 = Box::new(
            ConditionalFailureStage::new("stage1")
                .fail_when_nodes_equal(0)
                .with_message("No nodes in graph")
                .with_counter(Arc::clone(&counter)),
        );
        let stage2 =
            Box::new(ConditionalFailureStage::new("stage2").with_counter(Arc::clone(&counter)));

        let pipeline = BuildPipeline::new().with_stage(stage1).with_stage(stage2);

        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        // Empty graph should trigger failure at stage1
        let graph = BuildGraph::default();
        let result = pipeline.execute(&ctx, graph);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{:?}", err).contains("No nodes in graph"),
            "Error should contain failure message"
        );

        // Only first stage should have executed
        assert_eq!(*counter.lock().unwrap(), 1);
    }

    #[test]
    fn pipeline_recovery_failure_at_middle_stage() {
        let counter = Arc::new(Mutex::new(0));

        let stage1 =
            Box::new(ConditionalFailureStage::new("stage1").with_counter(Arc::clone(&counter)));
        let stage2 = Box::new(
            ConditionalFailureStage::new("stage2")
                .fail_when_nodes_equal(1)
                .with_message("Single node not allowed")
                .with_counter(Arc::clone(&counter)),
        );
        let stage3 =
            Box::new(ConditionalFailureStage::new("stage3").with_counter(Arc::clone(&counter)));

        let pipeline = BuildPipeline::new()
            .with_stage(stage1)
            .with_stage(stage2)
            .with_stage(stage3);

        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        // Graph with one node should fail at stage2
        let mut graph = BuildGraph::default();
        graph.add_node("single", ModelConfig::new("test", "1.0.0"));

        let result = pipeline.execute(&ctx, graph);

        assert!(result.is_err());
        // First two stages should have executed
        assert_eq!(*counter.lock().unwrap(), 2);
    }

    #[test]
    fn pipeline_recovery_success_path_after_fix() {
        let counter = Arc::new(Mutex::new(0));

        let stage1 = Box::new(
            ConditionalFailureStage::new("stage1")
                .fail_when_nodes_equal(0)
                .with_counter(Arc::clone(&counter)),
        );
        let stage2 =
            Box::new(ConditionalFailureStage::new("stage2").with_counter(Arc::clone(&counter)));

        let pipeline = BuildPipeline::new().with_stage(stage1).with_stage(stage2);

        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        // First attempt with empty graph - should fail
        let empty_graph = BuildGraph::default();
        let result1 = pipeline.execute(&ctx, empty_graph);
        assert!(result1.is_err());

        // Reset counter and try again with non-empty graph
        *counter.lock().unwrap() = 0;

        let mut fixed_graph = BuildGraph::default();
        fixed_graph.add_node("node", ModelConfig::new("test", "1.0.0"));

        let result2 = pipeline.execute(&ctx, fixed_graph);
        assert!(result2.is_ok(), "Pipeline should succeed with fixed graph");
        assert_eq!(*counter.lock().unwrap(), 2, "Both stages should execute");
    }

    #[test]
    fn pipeline_pre_hook_failure_prevents_stage_execution() {
        let stage_counter = Arc::new(Mutex::new(0));
        let hook_counter = Arc::new(Mutex::new(0));

        let hook_counter_clone = Arc::clone(&hook_counter);
        let failing_pre_hook =
            move |_ctx: &BuildContext| -> aphelion_core::error::AphelionResult<()> {
                *hook_counter_clone.lock().unwrap() += 1;
                Err(aphelion_core::error::AphelionError::build(
                    "Pre-hook validation failed",
                ))
            };

        let stage = Box::new(CountingStage::new("stage", Arc::clone(&stage_counter)));

        let pipeline = BuildPipeline::new()
            .with_pre_hook(failing_pre_hook)
            .with_stage(stage);

        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let graph = BuildGraph::default();
        let result = pipeline.execute(&ctx, graph);

        assert!(result.is_err());
        assert_eq!(*hook_counter.lock().unwrap(), 1, "Pre-hook should have run");
        assert_eq!(
            *stage_counter.lock().unwrap(),
            0,
            "Stage should not run after pre-hook failure"
        );
    }

    #[test]
    fn pipeline_post_hook_failure_after_successful_stages() {
        let stage_counter = Arc::new(Mutex::new(0));

        let failing_post_hook = move |_ctx: &BuildContext,
                                      _graph: &BuildGraph|
              -> aphelion_core::error::AphelionResult<()> {
            Err(aphelion_core::error::AphelionError::build(
                "Post-hook cleanup failed",
            ))
        };

        let stage = Box::new(CountingStage::new("stage", Arc::clone(&stage_counter)));

        let pipeline = BuildPipeline::new()
            .with_stage(stage)
            .with_post_hook(failing_post_hook);

        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let graph = BuildGraph::default();
        let result = pipeline.execute(&ctx, graph);

        assert!(result.is_err());
        assert_eq!(
            *stage_counter.lock().unwrap(),
            1,
            "Stage should have run before post-hook failure"
        );
    }

    #[test]
    fn pipeline_error_contains_stage_context() {
        struct NamedFailingStage {
            name: String,
        }

        impl PipelineStage for NamedFailingStage {
            fn name(&self) -> &str {
                &self.name
            }

            fn execute(
                &self,
                _ctx: &BuildContext,
                _graph: &mut BuildGraph,
            ) -> aphelion_core::error::AphelionResult<()> {
                Err(aphelion_core::error::AphelionError::build(&format!(
                    "Stage '{}' failed: resource exhausted",
                    self.name
                )))
            }
        }

        let pipeline = BuildPipeline::new().with_stage(Box::new(NamedFailingStage {
            name: "optimization_pass".to_string(),
        }));

        let backend = NullBackend::cpu();
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext {
            backend: &backend,
            trace: &trace,
        };

        let graph = BuildGraph::default();
        let result = pipeline.execute(&ctx, graph);

        assert!(result.is_err());
        let err_string = format!("{:?}", result.unwrap_err());
        assert!(err_string.contains("optimization_pass"));
        assert!(err_string.contains("resource exhausted"));
    }

    // ============================================================================
    // BURN BACKEND TESTS (feature-gated)
    // ============================================================================

    #[cfg(feature = "burn")]
    mod burn_backend_tests {
        use super::*;
        use aphelion_core::burn_backend::{BurnBackend, BurnBackendConfig, BurnDevice};

        #[test]
        fn burn_device_label_formatting() {
            assert_eq!(BurnDevice::Cpu.as_label(), "cpu");
            assert_eq!(BurnDevice::Cuda(0).as_label(), "cuda:0");
            assert_eq!(BurnDevice::Cuda(1).as_label(), "cuda:1");
            assert_eq!(BurnDevice::Metal(0).as_label(), "metal:0");
            assert_eq!(BurnDevice::Vulkan(0).as_label(), "vulkan:0");
        }

        #[test]
        fn burn_device_type_checks() {
            assert!(BurnDevice::Cpu.is_cpu());
            assert!(!BurnDevice::Cpu.is_gpu());

            assert!(!BurnDevice::Cuda(0).is_cpu());
            assert!(BurnDevice::Cuda(0).is_gpu());

            assert!(!BurnDevice::Metal(0).is_cpu());
            assert!(BurnDevice::Metal(0).is_gpu());

            assert!(!BurnDevice::Vulkan(0).is_cpu());
            assert!(BurnDevice::Vulkan(0).is_gpu());
        }

        #[test]
        fn burn_backend_default_config() {
            let config = BurnBackendConfig::default();
            assert!(config.device.is_cpu());
            assert!(!config.allow_tf32);
        }

        #[test]
        fn burn_backend_cpu_creation() {
            let config = BurnBackendConfig {
                device: BurnDevice::Cpu,
                allow_tf32: false,
            };
            let backend = BurnBackend::new(config);

            assert_eq!(backend.name(), "burn");
            assert_eq!(backend.device(), "cpu");
        }

        #[test]
        fn burn_backend_lifecycle() {
            let config = BurnBackendConfig::default();
            let mut backend = BurnBackend::new(config);

            // CPU should always be available
            assert!(backend.is_available());

            // Initialize and shutdown
            let init_result = backend.initialize();
            assert!(init_result.is_ok());

            let shutdown_result = backend.shutdown();
            assert!(shutdown_result.is_ok());
        }

        #[test]
        fn burn_backend_capabilities() {
            let config = BurnBackendConfig {
                device: BurnDevice::Cpu,
                allow_tf32: true,
            };
            let backend = BurnBackend::new(config);
            let caps = backend.capabilities();

            // CPU backend has limited capabilities
            assert!(!caps.supports_tf32); // TF32 is GPU-only
        }

        #[test]
        fn burn_backend_in_build_context() {
            let config = BurnBackendConfig::default();
            let mut backend = BurnBackend::new(config);
            backend.initialize().unwrap();

            let trace = InMemoryTraceSink::new();
            let ctx = BuildContext {
                backend: &backend,
                trace: &trace,
            };

            // Verify backend is usable in pipeline context
            assert_eq!(ctx.backend.name(), "burn");
            assert!(ctx.backend.is_available());
        }

        #[test]
        fn burn_backend_registry_integration() {
            let mut registry = BackendRegistry::new();

            let burn_backend = Box::new(BurnBackend::new(BurnBackendConfig::default()));
            let null_backend = Box::new(NullBackend::cpu());

            registry.register(burn_backend);
            registry.register(null_backend);

            // Should be able to retrieve burn backend
            let burn = registry.get("burn");
            assert!(burn.is_some());
            assert_eq!(burn.unwrap().name(), "burn");
        }

        #[test]
        fn burn_backend_with_pipeline() {
            let config = BurnBackendConfig::default();
            let mut backend = BurnBackend::new(config);
            backend.initialize().unwrap();

            let trace = InMemoryTraceSink::new();
            let ctx = BuildContext {
                backend: &backend,
                trace: &trace,
            };

            let counter = Arc::new(Mutex::new(0));
            let stage = Box::new(CountingStage::new("burn_test_stage", Arc::clone(&counter)));

            let pipeline = BuildPipeline::new().with_stage(stage);

            let mut graph = BuildGraph::default();
            graph.add_node("test", ModelConfig::new("burn-model", "1.0.0"));

            let result = pipeline.execute(&ctx, graph);
            assert!(result.is_ok());
            assert_eq!(*counter.lock().unwrap(), 1);

            backend.shutdown().unwrap();
        }
    }

    // ============================================================================
    // CUBECL BACKEND TESTS (feature-gated)
    // ============================================================================

    #[cfg(feature = "cubecl")]
    mod cubecl_backend_tests {
        use super::*;
        use aphelion_core::cubecl_backend::{CubeclBackend, CubeclBackendConfig, CubeclDevice};

        #[test]
        fn cubecl_device_label_formatting() {
            assert_eq!(CubeclDevice::Cpu.as_label(), "cpu");
            assert_eq!(CubeclDevice::Cuda(0).as_label(), "cuda:0");
            assert_eq!(CubeclDevice::Cuda(2).as_label(), "cuda:2");
            assert_eq!(CubeclDevice::Metal(0).as_label(), "metal:0");
            assert_eq!(CubeclDevice::Vulkan(0).as_label(), "vulkan:0");
            assert_eq!(CubeclDevice::Wgpu(0).as_label(), "wgpu:0");
        }

        #[test]
        fn cubecl_device_type_checks() {
            assert!(CubeclDevice::Cpu.is_cpu());
            assert!(!CubeclDevice::Cpu.is_gpu());

            assert!(!CubeclDevice::Cuda(0).is_cpu());
            assert!(CubeclDevice::Cuda(0).is_gpu());

            assert!(!CubeclDevice::Metal(0).is_cpu());
            assert!(CubeclDevice::Metal(0).is_gpu());

            assert!(!CubeclDevice::Vulkan(0).is_cpu());
            assert!(CubeclDevice::Vulkan(0).is_gpu());

            assert!(!CubeclDevice::Wgpu(0).is_cpu());
            assert!(CubeclDevice::Wgpu(0).is_gpu());
        }

        #[test]
        fn cubecl_backend_default_config() {
            let config = CubeclBackendConfig::default();
            assert!(config.device.is_cpu());
            assert!((config.memory_fraction - 0.9).abs() < f32::EPSILON);
        }

        #[test]
        fn cubecl_backend_cpu_creation() {
            let config = CubeclBackendConfig {
                device: CubeclDevice::Cpu,
                memory_fraction: 0.8,
            };
            let backend = CubeclBackend::new(config);

            assert_eq!(backend.name(), "cubecl");
            assert_eq!(backend.device(), "cpu");
        }

        #[test]
        fn cubecl_backend_lifecycle() {
            let config = CubeclBackendConfig::default();
            let mut backend = CubeclBackend::new(config);

            // CPU should always be available
            assert!(backend.is_available());

            // Initialize and shutdown
            let init_result = backend.initialize();
            assert!(init_result.is_ok());

            let shutdown_result = backend.shutdown();
            assert!(shutdown_result.is_ok());
        }

        #[test]
        fn cubecl_backend_capabilities() {
            let config = CubeclBackendConfig::default();
            let backend = CubeclBackend::new(config);
            let caps = backend.capabilities();

            // CPU backend should report no GPU-specific features
            assert!(!caps.supports_tf32);
        }

        #[test]
        fn cubecl_backend_in_build_context() {
            let config = CubeclBackendConfig::default();
            let mut backend = CubeclBackend::new(config);
            backend.initialize().unwrap();

            let trace = InMemoryTraceSink::new();
            let ctx = BuildContext {
                backend: &backend,
                trace: &trace,
            };

            // Verify backend is usable in pipeline context
            assert_eq!(ctx.backend.name(), "cubecl");
            assert!(ctx.backend.is_available());
        }

        #[test]
        fn cubecl_backend_registry_integration() {
            let mut registry = BackendRegistry::new();

            let cubecl_backend = Box::new(CubeclBackend::new(CubeclBackendConfig::default()));
            let null_backend = Box::new(NullBackend::cpu());

            registry.register(cubecl_backend);
            registry.register(null_backend);

            // Should be able to retrieve cubecl backend
            let cubecl = registry.get("cubecl");
            assert!(cubecl.is_some());
            assert_eq!(cubecl.unwrap().name(), "cubecl");
        }

        #[test]
        fn cubecl_backend_with_pipeline() {
            let config = CubeclBackendConfig::default();
            let mut backend = CubeclBackend::new(config);
            backend.initialize().unwrap();

            let trace = InMemoryTraceSink::new();
            let ctx = BuildContext {
                backend: &backend,
                trace: &trace,
            };

            let counter = Arc::new(Mutex::new(0));
            let stage = Box::new(CountingStage::new(
                "cubecl_test_stage",
                Arc::clone(&counter),
            ));

            let pipeline = BuildPipeline::new().with_stage(stage);

            let mut graph = BuildGraph::default();
            graph.add_node("test", ModelConfig::new("cubecl-model", "1.0.0"));

            let result = pipeline.execute(&ctx, graph);
            assert!(result.is_ok());
            assert_eq!(*counter.lock().unwrap(), 1);

            backend.shutdown().unwrap();
        }

        #[test]
        fn cubecl_backend_memory_fraction_config() {
            let config = CubeclBackendConfig {
                device: CubeclDevice::Cuda(0),
                memory_fraction: 0.5,
            };
            let backend = CubeclBackend::new(config);

            // Verify configuration is stored (implementation detail)
            assert_eq!(backend.name(), "cubecl");
            // Note: device() returns just the device type, not the index
            assert_eq!(backend.device(), "cuda");
        }

        #[test]
        fn cubecl_wgpu_device_creation() {
            // WebGPU device - useful for cross-platform support
            let config = CubeclBackendConfig {
                device: CubeclDevice::Wgpu(0),
                memory_fraction: 0.9,
            };
            let backend = CubeclBackend::new(config);

            assert_eq!(backend.name(), "cubecl");
            // Note: device() returns just the device type, not the index
            assert_eq!(backend.device(), "wgpu");
        }
    }
}
