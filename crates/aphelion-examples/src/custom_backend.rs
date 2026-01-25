//! Custom backend implementation example.
//!
//! This example demonstrates how to implement your own Backend trait
//! with custom device capabilities and features.
//!
//! Key concepts:
//! - Implement the Backend trait for custom hardware/framework integration
//! - Define custom DeviceCapabilities
//! - Use with BuildPipeline for model building with your backend
//! - Handle initialization and shutdown lifecycle

use aphelion_core::backend::{Backend, DeviceCapabilities};
use aphelion_core::config::ModelConfig;
use aphelion_core::diagnostics::InMemoryTraceSink;
use aphelion_core::error::AphelionResult;
use aphelion_core::graph::BuildGraph;
use aphelion_core::pipeline::{BuildContext, BuildPipeline};

/// A custom GPU backend implementation with advanced capabilities.
#[derive(Debug, Clone)]
pub struct CustomGpuBackend {
    name: String,
    device: String,
    initialized: std::sync::Arc<std::sync::Mutex<bool>>,
}

impl CustomGpuBackend {
    pub fn new(device: impl Into<String>) -> Self {
        Self {
            name: "custom_gpu".to_string(),
            device: device.into(),
            initialized: std::sync::Arc::new(std::sync::Mutex::new(false)),
        }
    }

    pub fn is_initialized(&self) -> bool {
        *self.initialized.lock().unwrap()
    }
}

impl Backend for CustomGpuBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn device(&self) -> &str {
        &self.device
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            supports_f16: true,
            supports_bf16: true,
            supports_tf32: true,
            max_memory_bytes: Some(8 * 1024 * 1024 * 1024), // 8GB
            compute_units: Some(5120),
        }
    }

    fn is_available(&self) -> bool {
        true
    }

    fn initialize(&mut self) -> AphelionResult<()> {
        *self.initialized.lock().unwrap() = true;
        println!("  -> Initialized {}", self.name);
        Ok(())
    }

    fn shutdown(&mut self) -> AphelionResult<()> {
        *self.initialized.lock().unwrap() = false;
        println!("  -> Shutdown {}", self.name);
        Ok(())
    }
}

/// A custom TPU backend with different capabilities.
#[derive(Debug, Clone)]
pub struct CustomTpuBackend {
    name: String,
    device: String,
}

impl CustomTpuBackend {
    pub fn new(device: impl Into<String>) -> Self {
        Self {
            name: "custom_tpu".to_string(),
            device: device.into(),
        }
    }
}

impl Backend for CustomTpuBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn device(&self) -> &str {
        &self.device
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            supports_f16: false,
            supports_bf16: true,
            supports_tf32: false,
            max_memory_bytes: Some(16 * 1024 * 1024 * 1024), // 16GB
            compute_units: Some(8192),
        }
    }

    fn is_available(&self) -> bool {
        true
    }

    fn initialize(&mut self) -> AphelionResult<()> {
        println!("  -> Initialized TPU device: {}", self.device);
        Ok(())
    }

    fn shutdown(&mut self) -> AphelionResult<()> {
        println!("  -> Shutdown TPU device");
        Ok(())
    }
}

/// Run the custom backend example.
///
/// This example demonstrates:
/// 1. Implementing custom backends with Backend trait
/// 2. Defining custom device capabilities
/// 3. Using custom backends with BuildPipeline
/// 4. Lifecycle management (initialization/shutdown)
pub fn run_example() -> AphelionResult<()> {
    println!("=== Custom Backend Example ===\n");

    // Create a GPU backend
    println!("1. Using Custom GPU Backend:");
    let mut gpu_backend = CustomGpuBackend::new("gpu:0");
    gpu_backend.initialize()?;

    let gpu_caps = gpu_backend.capabilities();
    println!("  GPU Capabilities:");
    println!("    - F16 Support: {}", gpu_caps.supports_f16);
    println!("    - BF16 Support: {}", gpu_caps.supports_bf16);
    println!("    - TF32 Support: {}", gpu_caps.supports_tf32);
    println!(
        "    - Max Memory: {} GB",
        gpu_caps.max_memory_bytes.unwrap_or(0) / (1024 * 1024 * 1024)
    );
    println!(
        "    - Compute Units: {}\n",
        gpu_caps.compute_units.unwrap_or(0)
    );

    // Create a model and build with GPU backend
    let config = ModelConfig::new("gpu_model", "1.0.0")
        .with_param("precision", serde_json::json!("float16"));

    let mut graph = BuildGraph::default();
    graph.add_node("inference_layer", config.clone());

    let trace_sink = InMemoryTraceSink::new();
    let ctx = BuildContext::new(&gpu_backend, &trace_sink);

    println!("  Building with GPU backend:");
    let pipeline = BuildPipeline::new()
        .with_pre_hook(|ctx| {
            println!(
                "    - Pre-build: Checking {} availability",
                ctx.backend.name()
            );
            Ok(())
        })
        .with_post_hook(|_ctx, graph| {
            println!("    - Post-build: Graph hash = {}", graph.stable_hash());
            Ok(())
        });

    pipeline.execute(&ctx, graph)?;
    gpu_backend.shutdown()?;

    println!("\n2. Using Custom TPU Backend:");
    let mut tpu_backend = CustomTpuBackend::new("tpu:0");
    tpu_backend.initialize()?;

    let tpu_caps = tpu_backend.capabilities();
    println!("  TPU Capabilities:");
    println!("    - F16 Support: {}", tpu_caps.supports_f16);
    println!("    - BF16 Support: {}", tpu_caps.supports_bf16);
    println!("    - TF32 Support: {}", tpu_caps.supports_tf32);
    println!(
        "    - Max Memory: {} GB",
        tpu_caps.max_memory_bytes.unwrap_or(0) / (1024 * 1024 * 1024)
    );
    println!(
        "    - Compute Units: {}\n",
        tpu_caps.compute_units.unwrap_or(0)
    );

    // Create a model and build with TPU backend
    let config = ModelConfig::new("tpu_model", "1.0.0")
        .with_param("precision", serde_json::json!("bfloat16"));

    let mut graph = BuildGraph::default();
    graph.add_node("tpu_layer", config);

    let trace_sink = InMemoryTraceSink::new();
    let ctx = BuildContext::new(&tpu_backend, &trace_sink);

    println!("  Building with TPU backend:");
    let pipeline = BuildPipeline::new();
    pipeline.execute(&ctx, graph)?;
    tpu_backend.shutdown()?;

    println!("\nCustom backend example completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_gpu_backend_capabilities() {
        let backend = CustomGpuBackend::new("gpu:0");
        let caps = backend.capabilities();
        assert!(caps.supports_f16);
        assert!(caps.supports_bf16);
        assert!(caps.supports_tf32);
    }

    #[test]
    fn test_custom_tpu_backend_capabilities() {
        let backend = CustomTpuBackend::new("tpu:0");
        let caps = backend.capabilities();
        assert!(!caps.supports_f16);
        assert!(caps.supports_bf16);
        assert!(!caps.supports_tf32);
    }

    #[test]
    fn test_custom_backend_example() {
        assert!(run_example().is_ok());
    }
}
