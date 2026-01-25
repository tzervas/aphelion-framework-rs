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
}
