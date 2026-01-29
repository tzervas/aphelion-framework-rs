//! WebAssembly/TypeScript bindings for the Aphelion AI Framework.
//!
//! This module provides WASM bindings for aphelion-core via wasm-bindgen,
//! enabling TypeScript/JavaScript developers to use Aphelion in browsers
//! and Node.js environments.
//!
//! ## Usage in TypeScript/JavaScript
//!
//! ```typescript
//! import init, { ModelConfig, BuildGraph, BuildPipeline } from 'aphelion-framework';
//!
//! await init();
//!
//! const config = new ModelConfig("transformer", "1.0.0");
//! config.withParam("d_model", 512);
//!
//! const graph = new BuildGraph();
//! const nodeId = graph.addNode("encoder", config);
//!
//! console.log(`Graph hash: ${graph.stableHash()}`);
//! ```

use wasm_bindgen::prelude::*;

mod backend;
mod config;
mod diagnostics;
mod graph;
mod pipeline;
mod validation;

pub use backend::*;
pub use config::*;
pub use diagnostics::*;
pub use graph::*;
pub use pipeline::*;
pub use validation::*;

/// Initialize the WASM module with better panic messages.
/// Call this once at startup for better error messages in development.
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Get the version of the aphelion-core library.
#[wasm_bindgen(js_name = getVersion)]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Check if the burn backend feature is enabled.
#[wasm_bindgen(js_name = hasBurn)]
pub fn has_burn() -> bool {
    cfg!(feature = "burn")
}

/// Check if the cubecl backend feature is enabled.
#[wasm_bindgen(js_name = hasCubecl)]
pub fn has_cubecl() -> bool {
    cfg!(feature = "cubecl")
}

/// Check if rust-ai-core integration is enabled.
#[wasm_bindgen(js_name = hasRustAiCore)]
pub fn has_rust_ai_core() -> bool {
    cfg!(feature = "rust-ai-core")
}
