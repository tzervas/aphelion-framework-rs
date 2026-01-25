//! Custom pipeline stages example.
//!
//! This example demonstrates how to:
//! - Implement custom PipelineStage trait
//! - Chain multiple stages together
//! - Add pre and post build hooks
//! - Use progress callbacks to track execution
//! - Skip stages conditionally
//!
//! Pipeline stages allow you to customize the build process with
//! validation, transformation, optimization, and other stages.

use aphelion_core::backend::NullBackend;
use aphelion_core::config::ModelConfig;
use aphelion_core::diagnostics::{InMemoryTraceSink, TraceEvent, TraceLevel};
use aphelion_core::error::AphelionResult;
use aphelion_core::graph::BuildGraph;
use aphelion_core::pipeline::{BuildContext, BuildPipeline, PipelineStage};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// A custom stage that validates the graph structure.
struct ValidateStructureStage;

impl PipelineStage for ValidateStructureStage {
    fn name(&self) -> &str {
        "validate_structure"
    }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph) -> AphelionResult<()> {
        ctx.trace.record(TraceEvent {
            id: "stage.validate_structure".to_string(),
            message: format!(
                "Validating graph structure: {} nodes, {} edges",
                graph.node_count(),
                graph.edge_count()
            ),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });

        if graph.node_count() == 0 {
            return Err(aphelion_core::error::AphelionError::Validation(
                "Graph must contain at least one node".to_string(),
            ));
        }

        Ok(())
    }
}

/// A custom stage that optimizes the graph.
struct OptimizeGraphStage;

impl PipelineStage for OptimizeGraphStage {
    fn name(&self) -> &str {
        "optimize_graph"
    }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph) -> AphelionResult<()> {
        ctx.trace.record(TraceEvent {
            id: "stage.optimize".to_string(),
            message: "Optimizing graph for execution".to_string(),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });
        Ok(())
    }
}

/// A custom stage that generates code or serializes the graph.
struct CodegenStage;

impl PipelineStage for CodegenStage {
    fn name(&self) -> &str {
        "codegen"
    }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph) -> AphelionResult<()> {
        let hash = graph.stable_hash();
        ctx.trace.record(TraceEvent {
            id: "stage.codegen".to_string(),
            message: format!("Generating code for graph (hash: {})", hash),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });
        Ok(())
    }
}

/// A custom stage that adds metadata to graph nodes.
struct MetadataStage {
    metadata_added: Arc<Mutex<usize>>,
}

impl MetadataStage {
    fn new() -> Self {
        Self {
            metadata_added: Arc::new(Mutex::new(0)),
        }
    }

    fn get_metadata_count(&self) -> usize {
        *self.metadata_added.lock().unwrap()
    }
}

impl PipelineStage for MetadataStage {
    fn name(&self) -> &str {
        "metadata"
    }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph) -> AphelionResult<()> {
        let mut count = self.metadata_added.lock().unwrap();
        for node in &mut graph.nodes {
            node.metadata.insert(
                "stage".to_string(),
                serde_json::json!("metadata"),
            );
            *count += 1;
        }

        ctx.trace.record(TraceEvent {
            id: "stage.metadata".to_string(),
            message: format!("Added metadata to {} nodes", *count),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });
        Ok(())
    }
}

/// Run the pipeline stages example.
///
/// This example demonstrates:
/// 1. Creating custom PipelineStage implementations
/// 2. Chaining multiple stages in sequence
/// 3. Using pre and post build hooks
/// 4. Progress tracking with callbacks
/// 5. Conditional stage skipping
pub fn run_example() -> AphelionResult<()> {
    println!("=== Pipeline Stages Example ===\n");

    // Create a model and graph
    let config = ModelConfig::new("pipeline_model", "1.0.0")
        .with_param("stage_count", serde_json::json!(4));

    let mut graph = BuildGraph::default();
    let node1 = graph.add_node("input", config.clone());
    let node2 = graph.add_node("processing", config.clone());
    let node3 = graph.add_node("output", config.clone());
    graph.add_edge(node1, node2);
    graph.add_edge(node2, node3);

    println!("Initial graph:");
    println!("  Nodes: {}", graph.node_count());
    println!("  Edges: {}", graph.edge_count());
    println!("  Hash: {}\n", graph.stable_hash());

    // Set up backend and tracing
    let backend = NullBackend::cpu();
    let trace_sink = InMemoryTraceSink::new();

    // Create a pipeline with multiple stages and hooks
    println!("Building pipeline with stages and hooks:");
    let metadata_stage = Box::new(MetadataStage::new());

    let pipeline = BuildPipeline::new()
        // Pre-build hook
        .with_pre_hook(|ctx| {
            println!("  [PRE-HOOK] Starting build process");
            println!("    Backend: {}", ctx.backend.name());
            println!("    Device: {}", ctx.backend.device());
            Ok(())
        })
        // Add stages
        .with_stage(Box::new(ValidateStructureStage))
        .with_stage(Box::new(OptimizeGraphStage))
        .with_stage(metadata_stage)
        .with_stage(Box::new(CodegenStage))
        // Post-build hook
        .with_post_hook(|_ctx, graph| {
            println!("  [POST-HOOK] Build completed");
            println!("    Final node count: {}", graph.node_count());
            println!("    Final edge count: {}", graph.edge_count());
            Ok(())
        })
        // Progress callback
        .with_progress(|stage_name, current, total| {
            let percent = (current as f32 / total as f32) * 100.0;
            println!("  [PROGRESS] {}/{} ({:.0}%) - {}", current, total, percent, stage_name);
        });

    let ctx = BuildContext {
        backend: &backend,
        trace: &trace_sink,
    };

    println!();
    pipeline.execute(&ctx, graph)?;

    // Print trace events
    let events = trace_sink.events();
    println!("\nTrace events ({} total):", events.len());
    for event in &events {
        println!("  [{}] {}", event.id, event.message);
    }

    // Example: Creating a pipeline with stage skipping
    println!("\n--- Pipeline with stage skipping ---\n");

    let config = ModelConfig::new("skip_example", "1.0.0");
    let mut graph = BuildGraph::default();
    graph.add_node("main", config);

    let trace_sink = InMemoryTraceSink::new();
    let ctx = BuildContext {
        backend: &backend,
        trace: &trace_sink,
    };

    let pipeline = BuildPipeline::new()
        .with_stage(Box::new(ValidateStructureStage))
        .with_stage(Box::new(OptimizeGraphStage))
        .with_stage(Box::new(CodegenStage))
        .with_skip_stage("optimize_graph")  // Skip optimization stage
        .with_progress(|stage_name, current, total| {
            println!("  [PROGRESS] {}/{} - {}", current, total, stage_name);
        });

    pipeline.execute(&ctx, graph)?;

    println!("\nPipeline stages example completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_stage() {
        let stage = MetadataStage::new();
        assert_eq!(stage.get_metadata_count(), 0);
    }

    #[test]
    fn test_pipeline_stages_example() {
        assert!(run_example().is_ok());
    }
}
