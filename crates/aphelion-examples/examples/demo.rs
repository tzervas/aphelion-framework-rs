//! Aphelion Framework Demo
//!
//! Run with: cargo run --example demo

use aphelion_core::backend::{Backend, NullBackend};
use aphelion_core::config::ModelConfig;
use aphelion_core::diagnostics::InMemoryTraceSink;
use aphelion_core::graph::BuildGraph;
use aphelion_core::pipeline::{BuildContext, BuildPipeline};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                 Aphelion Framework Demo");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Create model configuration
    let config = ModelConfig::new("transformer", "1.0.0")
        .with_param("d_model", serde_json::json!(512))
        .with_param("n_heads", serde_json::json!(8))
        .with_param("n_layers", serde_json::json!(6))
        .with_param("dropout", serde_json::json!(0.1));

    println!("📋 Model Configuration:");
    println!("   Name: {}", config.name);
    println!("   Version: {}", config.version);
    for (key, value) in &config.params {
        println!("   {} = {}", key, value);
    }
    println!();

    // Build graph
    let mut graph = BuildGraph::default();
    let encoder = graph.add_node("encoder", config.clone());
    let decoder = graph.add_node("decoder", config);
    graph.add_edge(encoder, decoder);

    println!("🔷 Build Graph:");
    println!("   Nodes: {}", graph.nodes.len());
    println!("   Edges: {}", graph.edges.len());
    println!("   Hash: {}\n", graph.stable_hash());

    // Execute pipeline
    let backend = NullBackend::cpu();
    let trace = InMemoryTraceSink::new();
    let ctx = BuildContext::new(&backend, &trace);

    println!("⚙️  Backend: {} ({})", backend.name(), backend.device());

    let pipeline = BuildPipeline::standard();
    let result = pipeline.execute(&ctx, graph).unwrap();

    println!("\n✅ Pipeline Execution Complete");
    println!("   Final Hash: {}", result.stable_hash());
    println!("   Trace Events: {}", trace.events().len());

    // Show trace events
    println!("\n📊 Trace Events:");
    for event in trace.events().iter().take(5) {
        println!("   [{:?}] {} - {}", event.level, event.id, event.message);
    }

    println!("\n═══════════════════════════════════════════════════════════════");
}
