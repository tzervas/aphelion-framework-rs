//! Python bindings for the Aphelion AI Framework.

use pyo3::prelude::*;

mod backend;
mod config;
mod diagnostics;
mod graph;
mod pipeline;
mod validation;

/// Aphelion Python module.
#[pymodule]
fn aphelion(m: &Bound<'_, PyModule>) -> PyResult<()> {
    config::register(m)?;
    graph::register(m)?;
    backend::register(m)?;
    diagnostics::register(m)?;
    pipeline::register(m)?;
    validation::register(m)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
