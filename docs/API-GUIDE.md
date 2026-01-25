# Aphelion API Guide

This guide covers common patterns and usage examples for the Aphelion framework. Each section includes complete, runnable code examples.

## Table of Contents

1. [Basic Model Building](#basic-model-building)
2. [Configuration Management](#configuration-management)
3. [Pipeline Usage](#pipeline-usage)
4. [Custom Backend Implementation](#custom-backend-implementation)
5. [Custom Pipeline Stages](#custom-pipeline-stages)
6. [Validation Patterns](#validation-patterns)
7. [Error Handling](#error-handling)
8. [Tracing and Diagnostics](#tracing-and-diagnostics)

## Basic Model Building

The fundamental workflow in Aphelion: create a config, build a graph, execute a pipeline.

```rust
use aphelion_core::prelude::*;

fn build_simple_model() -> AphelionResult<()> {
    // 1. Create configuration
    let config = ModelConfig::new("my-model", "1.0.0")
        .with_param("hidden_size", serde_json::json!(256))
        .with_param("num_layers", serde_json::json!(12));

    // 2. Build graph
    let mut graph = BuildGraph::default();
    let input_node = graph.add_node("input_layer", config.clone());
    let hidden_node = graph.add_node("hidden_layer", config.clone());
    let output_node = graph.add_node("output_layer", config);

    // Add edges to define data flow
    graph.add_edge(input_node, hidden_node);
    graph.add_edge(hidden_node, output_node);

    // 3. Set up execution context
    let backend = NullBackend::cpu();
    let trace = InMemoryTraceSink::new();
    let ctx = BuildContext::new(&backend, &trace);

    // 4. Execute pipeline
    let pipeline = BuildPipeline::new();
    let result = pipeline.execute(&ctx, graph)?;

    // 5. Get deterministic hash
    let hash = result.stable_hash();
    println!("Model graph hash: {}", hash);

    Ok(())
}
```

## Configuration Management

### Creating Configurations

```rust
use aphelion_core::config::ModelConfig;

// Simple config
let config = ModelConfig::new("gpt-style", "1.0.0");

// Config with parameters (builder pattern)
let config = ModelConfig::new("transformer", "2.1.0")
    .with_param("vocab_size", serde_json::json!(50000))
    .with_param("d_model", serde_json::json!(768))
    .with_param("n_heads", serde_json::json!(12))
    .with_param("n_layers", serde_json::json!(12))
    .with_param("dropout", serde_json::json!(0.1))
    .with_param("max_seq_len", serde_json::json!(512));
```

### Type-Safe Parameter Access

```rust
use aphelion_core::config::ModelConfig;

let config = ModelConfig::new("model", "1.0.0")
    .with_param("learning_rate", serde_json::json!(0.001))
    .with_param("batch_size", serde_json::json!(32));

// Type-safe parameter access with defaults
let lr: f64 = config.param_or("learning_rate", 0.01)?;

let batch_size: u32 = config.param_or("batch_size", 16)?;
```

### Using ModelConfigBuilder

```rust
use aphelion_core::config::ModelConfigBuilder;

// Using the builder for complex configurations
let config = ModelConfigBuilder::new()
    .name("bert-base")
    .version("1.0.0")
    .param("hidden_size", serde_json::json!(768))
    .param("num_layers", serde_json::json!(12))
    .param("num_heads", serde_json::json!(12))
    .build()
    .expect("valid config");
```

## Pipeline Usage

### Standard Pipeline

The simplest pipeline with no custom stages:

```rust
use aphelion_core::prelude::*;

let pipeline = BuildPipeline::new();
let result = pipeline.execute(&ctx, graph)?;
```

### Pipeline with Progress Tracking

```rust
use aphelion_core::pipeline::BuildPipeline;

let pipeline = BuildPipeline::new()
    .with_progress(|stage_name, current, total| {
        println!("Progress: [{}/{}] {}", current, total, stage_name);
    });

pipeline.execute(&ctx, graph)?;
```

### Pipeline with Hooks

Pre and post hooks run before and after all stages:

```rust
use aphelion_core::pipeline::BuildPipeline;

let pipeline = BuildPipeline::new()
    .with_pre_hook(|ctx| {
        println!("Starting build on backend: {}", ctx.backend.name());
        Ok(())
    })
    .with_post_hook(|ctx, graph| {
        println!("Build complete. Graph hash: {}", graph.stable_hash());
        println!("Final node count: {}", graph.node_count());
        Ok(())
    });

pipeline.execute(&ctx, graph)?;
```

### Pipeline Presets

Aphelion provides preset pipelines for common workflows:

```rust
use aphelion_core::pipeline::BuildPipeline;

// Standard pipeline (validation + basic stages)
let pipeline = BuildPipeline::standard();

// Training pipeline (validation + optimization + instrumentation)
let pipeline = BuildPipeline::for_training();

// Inference pipeline (validation + optimization for speed)
let pipeline = BuildPipeline::for_inference();

pipeline.execute(&ctx, graph)?;
```

### Custom Stages

Add custom stages to the pipeline:

```rust
use aphelion_core::pipeline::{BuildPipeline, PipelineStage};

struct OptimizationStage;

impl PipelineStage for OptimizationStage {
    fn name(&self) -> &str { "optimization" }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph)
        -> AphelionResult<()>
    {
        // Optimization logic here
        ctx.trace.info("optimization", "Optimizing graph structure");
        Ok(())
    }
}

let pipeline = BuildPipeline::new()
    .with_stage(Box::new(OptimizationStage));

pipeline.execute(&ctx, graph)?;
```

### Stage Skipping

Skip specific stages conditionally:

```rust
let pipeline = BuildPipeline::new()
    .with_stage(Box::new(ValidationStage))
    .with_stage(Box::new(OptimizationStage))
    .with_stage(Box::new(CodegenStage))
    .with_skip_stage("optimization");  // Skip optimization

pipeline.execute(&ctx, graph)?;
```

## Custom Backend Implementation

### Basic Custom Backend

```rust
use aphelion_core::backend::{Backend, DeviceCapabilities, MemoryInfo};
use aphelion_core::error::AphelionResult;

#[derive(Clone)]
struct MyGpuBackend {
    device_id: u32,
}

impl MyGpuBackend {
    fn new(device_id: u32) -> Self {
        Self { device_id }
    }
}

impl Backend for MyGpuBackend {
    fn name(&self) -> &str {
        "my_gpu"
    }

    fn device(&self) -> &str {
        "cuda:0"
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            supports_f16: true,
            supports_bf16: true,
            supports_tf32: true,
            max_memory_bytes: Some(8 * 1024 * 1024 * 1024), // 8GB
            compute_units: Some(2560),
        }
    }

    fn is_available(&self) -> bool {
        // Check if GPU is available
        true
    }

    fn initialize(&mut self) -> AphelionResult<()> {
        // Initialize GPU context
        println!("Initializing GPU device {}", self.device_id);
        Ok(())
    }

    fn shutdown(&mut self) -> AphelionResult<()> {
        // Clean up GPU resources
        println!("Shutting down GPU device {}", self.device_id);
        Ok(())
    }

    fn memory_info(&self) -> Option<MemoryInfo> {
        Some(MemoryInfo {
            total_bytes: 8 * 1024 * 1024 * 1024,
            used_bytes: 2 * 1024 * 1024 * 1024,
            free_bytes: 6 * 1024 * 1024 * 1024,
        })
    }
}
```

### Using Custom Backend

```rust
use aphelion_core::prelude::*;

fn main() -> AphelionResult<()> {
    let mut backend = MyGpuBackend::new(0);
    backend.initialize()?;

    // Check capabilities
    let caps = backend.capabilities();
    println!("FP16 support: {}", caps.supports_f16);
    println!("Max memory: {} GB",
        caps.max_memory_bytes.unwrap_or(0) / (1024 * 1024 * 1024));

    // Use backend in pipeline
    let trace = InMemoryTraceSink::new();
    let ctx = BuildContext::new(&backend, &trace);

    let config = ModelConfig::new("gpu-model", "1.0.0");
    let mut graph = BuildGraph::default();
    graph.add_node("layer", config);

    let pipeline = BuildPipeline::new();
    pipeline.execute(&ctx, graph)?;

    backend.shutdown()?;
    Ok(())
}
```

### Backend Registry and Auto-Detection

```rust
use aphelion_core::backend::{BackendRegistry, NullBackend};

// Create a registry and auto-detect best available backend
let mut registry = BackendRegistry::new();
registry.register(Box::new(NullBackend::cpu()));
// registry.register(Box::new(MyGpuBackend::new(0)));

if let Some(backend) = registry.auto_detect() {
    println!("Using backend: {}", backend.name());
}
```

## Custom Pipeline Stages

### Validation Stage

```rust
use aphelion_core::pipeline::{PipelineStage, BuildContext};
use aphelion_core::graph::BuildGraph;
use aphelion_core::error::{AphelionResult, AphelionError};

struct GraphValidationStage {
    min_nodes: usize,
}

impl GraphValidationStage {
    fn new(min_nodes: usize) -> Self {
        Self { min_nodes }
    }
}

impl PipelineStage for GraphValidationStage {
    fn name(&self) -> &str {
        "graph_validation"
    }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph)
        -> AphelionResult<()>
    {
        ctx.trace.info(
            "graph_validation",
            &format!("Validating graph with {} nodes", graph.node_count())
        );

        if graph.node_count() < self.min_nodes {
            return Err(AphelionError::Validation(
                format!("Graph must have at least {} nodes", self.min_nodes)
            ));
        }

        ctx.trace.info("graph_validation", "Validation passed");
        Ok(())
    }
}

// Usage
let pipeline = BuildPipeline::new()
    .with_stage(Box::new(GraphValidationStage::new(1)));
```

### Transformation Stage

```rust
use aphelion_core::pipeline::{PipelineStage, BuildContext};
use aphelion_core::graph::BuildGraph;
use aphelion_core::error::AphelionResult;

struct MetadataInjectionStage;

impl PipelineStage for MetadataInjectionStage {
    fn name(&self) -> &str {
        "metadata_injection"
    }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph)
        -> AphelionResult<()>
    {
        // Add metadata to all nodes
        for node in &mut graph.nodes {
            node.metadata.insert(
                "injected_timestamp".to_string(),
                serde_json::json!(std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_else(|_| std::time::Duration::from_secs(0))
                    .as_secs())
            );
            node.metadata.insert(
                "backend".to_string(),
                serde_json::json!(ctx.backend.name())
            );
        }

        ctx.trace.info(
            "metadata_injection",
            &format!("Injected metadata into {} nodes", graph.node_count())
        );

        Ok(())
    }
}
```

### Code Generation Stage

```rust
use aphelion_core::pipeline::{PipelineStage, BuildContext};
use aphelion_core::graph::BuildGraph;
use aphelion_core::error::AphelionResult;
use std::fs::File;
use std::io::Write;

struct CodegenStage {
    output_path: String,
}

impl CodegenStage {
    fn new(output_path: impl Into<String>) -> Self {
        Self {
            output_path: output_path.into(),
        }
    }
}

impl PipelineStage for CodegenStage {
    fn name(&self) -> &str {
        "codegen"
    }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph)
        -> AphelionResult<()>
    {
        let hash = graph.stable_hash();

        ctx.trace.info(
            "codegen",
            &format!("Generating code for graph {}", hash)
        );

        // Generate code (simplified example)
        let code = format!(
            "// Generated code for graph {}\n// Nodes: {}\n// Edges: {}\n",
            hash,
            graph.node_count(),
            graph.edge_count()
        );

        // Write to file
        let mut file = File::create(&self.output_path)
            .map_err(|e| AphelionError::Io(e))?;
        file.write_all(code.as_bytes())
            .map_err(|e| AphelionError::Io(e))?;

        ctx.trace.info("codegen", &format!("Code written to {}", self.output_path));

        Ok(())
    }
}
```

## Validation Patterns

### Using Built-in Validators

```rust
use aphelion_core::validation::{
    ConfigValidator, NameValidator, VersionValidator, CompositeValidator
};
use aphelion_core::config::ModelConfig;

// Single validator
let config = ModelConfig::new("my-model", "1.0.0");
let result = NameValidator.validate(&config);
assert!(result.is_ok());

// Composite validator
let validator = CompositeValidator::new()
    .with_validator(Box::new(NameValidator))
    .with_validator(Box::new(VersionValidator));

let result = validator.validate(&config);
assert!(result.is_ok());
```

### Custom Parameter Validator

```rust
use aphelion_core::validation::{ConfigValidator, ValidationError};
use aphelion_core::config::ModelConfig;

struct HyperparamValidator {
    min_lr: f64,
    max_lr: f64,
}

impl HyperparamValidator {
    fn new(min_lr: f64, max_lr: f64) -> Self {
        Self { min_lr, max_lr }
    }
}

impl ConfigValidator for HyperparamValidator {
    fn validate(&self, config: &ModelConfig)
        -> Result<(), Vec<ValidationError>>
    {
        let mut errors = Vec::new();

        if let Some(lr) = config.params.get("learning_rate")
            .and_then(|v| v.as_f64())
        {
            if lr < self.min_lr || lr > self.max_lr {
                errors.push(ValidationError::new(
                    "learning_rate",
                    format!("Must be between {} and {}", self.min_lr, self.max_lr)
                ));
            }
        } else {
            errors.push(ValidationError::new(
                "learning_rate",
                "Required parameter missing"
            ));
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

// Usage
let validator = HyperparamValidator::new(0.0001, 0.1);
let config = ModelConfig::new("model", "1.0.0")
    .with_param("learning_rate", serde_json::json!(0.001));

validator.validate(&config)?;
```

### Validation Pipeline Integration

```rust
use aphelion_core::pipeline::{BuildPipeline, PipelineStage};
use aphelion_core::validation::{ConfigValidator, CompositeValidator, NameValidator};

struct ConfigValidationStage {
    validator: CompositeValidator,
}

impl ConfigValidationStage {
    fn new() -> Self {
        Self {
            validator: CompositeValidator::new()
                .with_validator(Box::new(NameValidator))
                .with_validator(Box::new(VersionValidator)),
        }
    }
}

impl PipelineStage for ConfigValidationStage {
    fn name(&self) -> &str { "config_validation" }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph)
        -> AphelionResult<()>
    {
        for node in &graph.nodes {
            self.validator.validate(&node.config)
                .map_err(|errors| {
                    let msg = errors.iter()
                        .map(|e| e.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    AphelionError::InvalidConfig(msg)
                })?;
        }

        ctx.trace.info("config_validation", "All configs validated");
        Ok(())
    }
}
```

## Error Handling

### Basic Error Handling

All Aphelion operations return `AphelionResult<T>`:

```rust
use aphelion_core::error::{AphelionResult, AphelionError};

fn build_model() -> AphelionResult<BuildGraph> {
    let config = ModelConfig::new("model", "1.0.0");
    let mut graph = BuildGraph::default();

    if config.name.is_empty() {
        return Err(AphelionError::InvalidConfig(
            "Model name cannot be empty".to_string()
        ));
    }

    graph.add_node("layer", config);
    Ok(graph)
}

// Usage
match build_model() {
    Ok(graph) => println!("Built graph with {} nodes", graph.node_count()),
    Err(e) => eprintln!("Build failed: {}", e),
}
```

### Error Types

```rust
use aphelion_core::error::AphelionError;

// Different error types
let err = AphelionError::InvalidConfig("Bad config".to_string());
let err = AphelionError::Backend("GPU not available".to_string());
let err = AphelionError::Build("Pipeline stage failed".to_string());
let err = AphelionError::Validation("Invalid parameter".to_string());
let err = AphelionError::Serialization("JSON error".to_string());
let err = AphelionError::Graph("Cycle detected".to_string());
```

### Error Context and Chaining

Errors include context about where they occurred:

```rust
use aphelion_core::error::{AphelionError, AphelionResult};

fn validate_config(config: &ModelConfig) -> AphelionResult<()> {
    config.params.get("required_field")
        .ok_or_else(|| AphelionError::InvalidConfig(
            "Missing required_field in config".to_string()
        ))?;
    Ok(())
}

fn build_with_context() -> AphelionResult<()> {
    let config = ModelConfig::new("model", "1.0.0");

    validate_config(&config)
        .map_err(|e| AphelionError::Build(
            format!("Validation failed during build: {}", e)
        ))?;

    Ok(())
}
```

### Error Recovery

```rust
use aphelion_core::prelude::*;

fn build_with_fallback() -> AphelionResult<BuildGraph> {
    let config = ModelConfig::new("model", "1.0.0");
    let mut graph = BuildGraph::default();

    // Try with validation
    let backend = NullBackend::cpu();
    let trace = InMemoryTraceSink::new();
    let ctx = BuildContext::new(&backend, &trace);

    let pipeline = BuildPipeline::standard();

    match pipeline.execute(&ctx, graph.clone()) {
        Ok(g) => Ok(g),
        Err(AphelionError::Validation(_)) => {
            // Fall back to basic pipeline without validation
            eprintln!("Validation failed, using basic pipeline");
            let basic_pipeline = BuildPipeline::new();
            basic_pipeline.execute(&ctx, graph)
        }
        Err(e) => Err(e),
    }
}
```

## Tracing and Diagnostics

### Basic Tracing

```rust
use aphelion_core::diagnostics::{InMemoryTraceSink, TraceEvent, TraceLevel};
use std::time::SystemTime;

let trace_sink = InMemoryTraceSink::new();

// Record events manually
trace_sink.record(TraceEvent {
    id: "build.start".to_string(),
    message: "Starting model build".to_string(),
    timestamp: SystemTime::now(),
    level: TraceLevel::Info,
    span_id: None,
    trace_id: None,
});

// Get recorded events
let events = trace_sink.events();
for event in events {
    println!("[{}] {}: {}", event.level, event.id, event.message);
}
```

### Helper Methods

Use the `TraceSinkExt` trait for convenience:

```rust
use aphelion_core::diagnostics::{InMemoryTraceSink, TraceSinkExt};

let trace = InMemoryTraceSink::new();

// Helper methods (cleaner than creating TraceEvent manually)
trace.debug("stage.init", "Initializing stage");
trace.info("stage.execute", "Executing stage");
trace.warn("stage.perf", "Stage took longer than expected");
trace.error("stage.fail", "Stage failed");
```

### Trace Export

```rust
use aphelion_core::diagnostics::InMemoryTraceSink;

let trace = InMemoryTraceSink::new();

// ... build process with trace recording ...

// Export to JSON
let events = trace.events();
let json = serde_json::to_string_pretty(&events)?;
println!("{}", json);

// Or write to file
std::fs::write("trace.json", json)?;
```

### Distributed Tracing

```rust
use aphelion_core::diagnostics::{TraceEvent, TraceLevel, InMemoryTraceSink};
use std::time::SystemTime;

let trace = InMemoryTraceSink::new();

// Create events with span/trace IDs for distributed tracing
let trace_id = uuid::Uuid::new_v4().to_string();
let span_id = uuid::Uuid::new_v4().to_string();

trace.record(TraceEvent {
    id: "pipeline.stage1".to_string(),
    message: "Stage 1 executing".to_string(),
    timestamp: SystemTime::now(),
    level: TraceLevel::Info,
    span_id: Some(span_id.clone()),
    trace_id: Some(trace_id.clone()),
});

trace.record(TraceEvent {
    id: "pipeline.stage2".to_string(),
    message: "Stage 2 executing".to_string(),
    timestamp: SystemTime::now(),
    level: TraceLevel::Info,
    span_id: Some(span_id),
    trace_id: Some(trace_id),
});
```

### Custom Trace Sink

```rust
use aphelion_core::diagnostics::{TraceSink, TraceEvent};
use std::sync::{Arc, Mutex};

struct FileTraceSink {
    file_path: String,
    buffer: Arc<Mutex<Vec<TraceEvent>>>,
}

impl FileTraceSink {
    fn new(file_path: impl Into<String>) -> Self {
        Self {
            file_path: file_path.into(),
            buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn flush(&self) -> std::io::Result<()> {
        let events = self.buffer.lock().unwrap();
        let json = serde_json::to_string_pretty(&*events)?;
        std::fs::write(&self.file_path, json)?;
        Ok(())
    }
}

impl TraceSink for FileTraceSink {
    fn record(&self, event: TraceEvent) {
        self.buffer.lock().unwrap().push(event);
    }
}

// Usage
let trace = FileTraceSink::new("trace.json");
// ... use trace in pipeline ...
trace.flush()?;

// Or use the built-in JsonExporter
use aphelion_core::export::JsonExporter;

let exporter = JsonExporter::new();
// ... use exporter in pipeline ...
let mut file = std::fs::File::create("trace.json")?;
exporter.write_to(&mut file)?;
```

## Best Practices

### 1. Always Use Type-Safe Configuration

```rust
// Good: Type-safe parameter access
let hidden_size: u32 = config.param("hidden_size").unwrap_or(256);

// Avoid: Unsafe unwrapping
let hidden_size = config.params.get("hidden_size").unwrap().as_u64().unwrap();
```

### 2. Validate Early

```rust
// Good: Validate before expensive operations
let validator = CompositeValidator::new()
    .with_validator(Box::new(NameValidator))
    .with_validator(Box::new(VersionValidator));
validator.validate(&config)?;

// Then proceed with expensive graph building
let mut graph = BuildGraph::default();
// ...
```

### 3. Use Presets for Common Patterns

```rust
// Good: Use presets for standard workflows
let pipeline = BuildPipeline::standard();

// Avoid: Manually recreating common patterns
let pipeline = BuildPipeline::new()
    .with_stage(Box::new(ValidationStage))
    .with_stage(Box::new(HashingStage));
```

### 4. Handle Errors Explicitly

```rust
// Good: Explicit error handling
match pipeline.execute(&ctx, graph) {
    Ok(result) => process_result(result),
    Err(AphelionError::Validation(msg)) => handle_validation_error(msg),
    Err(e) => handle_other_error(e),
}

// Avoid: Silent failures
let result = pipeline.execute(&ctx, graph).ok();
```

### 5. Use Trace Events for Debugging

```rust
// Good: Trace important operations
ctx.trace.info("model.build", &format!("Building model: {}", config.name));
ctx.trace.debug("graph.hash", &format!("Graph hash: {}", graph.stable_hash()));

// This helps debug issues in production
```

## See Also

- [Architecture Guide](ARCHITECTURE.md) - System design and internals
- [Examples](../crates/aphelion-examples/src/) - Runnable code examples
- [API Documentation](https://docs.rs/aphelion-core) - Full API reference
