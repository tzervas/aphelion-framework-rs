//! Aphelion Framework - Ternary Acceleration Demo
//!
//! Run with: cargo run --example tritter_demo --features tritter-accel
//!
//! Demonstrates BitNet b1.58 ternary inference and VSA gradient compression.

#[cfg(feature = "tritter-accel")]
use aphelion_core::acceleration::{AccelerationStage, InferenceAccelConfig, TrainingAccelConfig};
use aphelion_core::backend::{Backend, NullBackend};
use aphelion_core::config::ModelConfig;
use aphelion_core::diagnostics::InMemoryTraceSink;
use aphelion_core::graph::BuildGraph;
use aphelion_core::pipeline::{BuildContext, BuildPipeline, HashingStage, ValidationStage};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("          Aphelion Framework - Ternary Acceleration");
    println!("═══════════════════════════════════════════════════════════════\n");

    #[cfg(not(feature = "tritter-accel"))]
    {
        println!("⚠️  tritter-accel feature not enabled!");
        println!("   Run with: cargo run --example tritter_demo --features tritter-accel");
        return;
    }

    #[cfg(feature = "tritter-accel")]
    {
        // Setup
        let backend = NullBackend::new("cuda:0");
        let trace = InMemoryTraceSink::new();
        let ctx = BuildContext::new(&backend, &trace);

        println!("⚙️  Backend: {} ({})\n", backend.name(), backend.device());

        // ═══════════════════════════════════════════════════════════════
        // TERNARY INFERENCE DEMO
        // ═══════════════════════════════════════════════════════════════
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("🔷 BitNet B1.58 Ternary Inference");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let config = ModelConfig::new("llama-7b-ternary", "1.0.0")
            .with_param("d_model", serde_json::json!(4096))
            .with_param("n_heads", serde_json::json!(32))
            .with_param("n_layers", serde_json::json!(32));

        let mut graph = BuildGraph::default();
        graph.add_node("model", config);

        // Configure ternary inference
        let inference_config = InferenceAccelConfig::new(32) // batch size 32
            .with_kv_cache(2048); // max seq len

        let pipeline = BuildPipeline::new()
            .with_stage(Box::new(ValidationStage))
            .with_stage(Box::new(AccelerationStage::with_inference_config(
                inference_config,
            )))
            .with_stage(Box::new(HashingStage));

        let result = pipeline.execute(&ctx, graph).unwrap();

        println!("📊 Inference Configuration Applied:");
        for node in &result.nodes {
            if let Some(mode) = node.metadata.get("accel.mode") {
                println!("   Mode: {}", mode);
            }
            if let Some(ternary) = node.metadata.get("accel.ternary_layers") {
                println!("   Ternary Layers: {} (16x memory reduction)", ternary);
            }
            if let Some(batch) = node.metadata.get("accel.batch_size") {
                println!("   Batch Size: {}", batch);
            }
            if let Some(kv) = node.metadata.get("accel.kv_cache") {
                println!("   KV Cache: {}", kv);
            }
            if let Some(seq) = node.metadata.get("accel.max_seq_len") {
                println!("   Max Seq Len: {}", seq);
            }
        }

        println!("\n   Memory Savings:");
        println!("   - Float32 weights: ~28 GB (7B × 4 bytes)");
        println!("   - Ternary packed:  ~1.75 GB (7B × 2 bits / 8)");
        println!("   - Reduction: 16x\n");

        // ═══════════════════════════════════════════════════════════════
        // VSA GRADIENT COMPRESSION DEMO
        // ═══════════════════════════════════════════════════════════════
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("🔷 VSA Gradient Compression Training");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let config = ModelConfig::new("llama-7b-finetune", "1.0.0")
            .with_param("d_model", serde_json::json!(4096))
            .with_param("learning_rate", serde_json::json!(2e-5));

        let mut graph = BuildGraph::default();
        graph.add_node("model", config);

        // Configure VSA training
        let training_config = TrainingAccelConfig::new(0.1) // 10x compression
            .with_seed(42)
            .with_mixed_precision();

        let pipeline = BuildPipeline::new()
            .with_stage(Box::new(ValidationStage))
            .with_stage(Box::new(AccelerationStage::with_training_config(
                training_config,
            )))
            .with_stage(Box::new(HashingStage));

        let result = pipeline.execute(&ctx, graph).unwrap();

        println!("📊 Training Configuration Applied:");
        for node in &result.nodes {
            if let Some(mode) = node.metadata.get("accel.mode") {
                println!("   Mode: {}", mode);
            }
            if let Some(ratio) = node.metadata.get("accel.compression_ratio") {
                println!(
                    "   Compression Ratio: {} ({}x reduction)",
                    ratio,
                    (1.0 / ratio.as_f64().unwrap_or(1.0)) as i32
                );
            }
            if let Some(det) = node.metadata.get("accel.deterministic") {
                println!("   Deterministic: {}", det);
            }
            if let Some(seed) = node.metadata.get("accel.seed") {
                println!("   Seed: {}", seed);
            }
            if let Some(mp) = node.metadata.get("accel.mixed_precision") {
                println!("   Mixed Precision: {}", mp);
            }
        }

        println!("\n   Bandwidth Savings (distributed training):");
        println!("   - Full gradients: 28 GB per sync");
        println!("   - VSA compressed: 2.8 GB per sync");
        println!("   - Accuracy loss: <5%\n");

        // Summary
        println!("═══════════════════════════════════════════════════════════════");
        println!("✅ Acceleration Stages Applied Successfully");
        println!("═══════════════════════════════════════════════════════════════");
    }
}
