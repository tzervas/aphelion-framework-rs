//! Basic usage example demonstrating simple model building.
//!
//! This example shows the fundamental workflow:
//! - Create a ModelConfig with basic parameters
//! - Build a BuildGraph with nodes and edges
//! - Execute through BuildPipeline
//! - Export traces to JSON for inspection
//!
//! This is the recommended starting point for understanding the aphelion framework.

use aphelion_core::backend::NullBackend;
use aphelion_core::config::ModelConfig;
use aphelion_core::diagnostics::{InMemoryTraceSink, TraceEvent, TraceLevel};
use aphelion_core::error::AphelionResult;
use aphelion_core::graph::BuildGraph;
use aphelion_core::pipeline::{BuildContext, BuildPipeline, PipelineStage};
use std::time::SystemTime;

/// A simple custom pipeline stage that logs model information.
struct ModelInfoStage;

impl PipelineStage for ModelInfoStage {
    fn name(&self) -> &str {
        "model_info"
    }

    fn execute(&self, ctx: &BuildContext, graph: &mut BuildGraph) -> AphelionResult<()> {
        ctx.trace.record(TraceEvent {
            id: "stage.model_info".to_string(),
            message: format!(
                "Graph contains {} nodes and {} edges",
                graph.node_count(),
                graph.edge_count()
            ),
            timestamp: SystemTime::now(),
            level: TraceLevel::Info,
            span_id: None,
            trace_id: None,
        });
        Ok(())
    }
}

/// Run the basic usage example.
///
/// This example demonstrates:
/// 1. Creating a ModelConfig with parameters
/// 2. Building a BuildGraph with nodes
/// 3. Executing through BuildPipeline with custom stages
/// 4. Collecting trace events for diagnostics
pub fn run_example() -> AphelionResult<()> {
    println!("=== Basic Usage Example ===\n");

    // Step 1: Create a ModelConfig
    let config = ModelConfig::new("my_model", "1.0.0")
        .with_param("hidden_size", serde_json::json!(256))
        .with_param("layers", serde_json::json!(4))
        .with_param("dropout", serde_json::json!(0.1));

    println!("Created ModelConfig:");
    println!("  Name: {}", config.name);
    println!("  Version: {}", config.version);
    println!("  Parameters: {:?}\n", config.params);

    // Step 2: Build a simple graph
    let mut graph = BuildGraph::default();
    let node_id = graph.add_node("main_layer", config.clone());
    graph.add_edge(node_id, node_id);

    println!("Built BuildGraph:");
    println!("  Nodes: {}", graph.node_count());
    println!("  Edges: {}", graph.edge_count());
    println!("  Hash: {}\n", graph.stable_hash());

    // Step 3: Set up backend and tracing
    let backend = NullBackend::cpu();
    let trace_sink = InMemoryTraceSink::new();

    // Step 4: Create and execute pipeline
    let pipeline = BuildPipeline::new()
        .with_stage(Box::new(ModelInfoStage))
        .with_progress(|stage_name, current, total| {
            println!("  Progress: [{}/{}] {}", current, total, stage_name);
        });

    let ctx = BuildContext::new(&backend, &trace_sink);

    println!("Executing BuildPipeline:");
    let result_graph = pipeline.execute(&ctx, graph)?;

    println!("\nPipeline completed successfully!");
    println!("  Result nodes: {}", result_graph.node_count());
    println!("  Result edges: {}", result_graph.edge_count());

    // Step 5: Export and inspect traces
    let events = trace_sink.events();
    println!("\nCollected {} trace events:", events.len());
    for event in &events {
        println!(
            "  [{}] {}: {}",
            if event.level == TraceLevel::Info {
                "INFO"
            } else {
                "DEBUG"
            },
            event.id,
            event.message
        );
    }

    // Step 6: Export traces to JSON
    let trace_json =
        serde_json::to_string_pretty(&events).unwrap_or_else(|_| "Failed to serialize".to_string());
    println!("\nTrace JSON export:\n{}\n", trace_json);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_usage_example() {
        assert!(run_example().is_ok());
    }
}
