//! Python bindings for graph types.

use pyo3::prelude::*;
use std::collections::HashMap;

use aphelion_core::graph::{BuildGraph, GraphNode, NodeId};

use crate::config::PyModelConfig;

/// Unique identifier for a graph node.
#[pyclass(name = "NodeId")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyNodeId {
    pub(crate) inner: NodeId,
}

#[pymethods]
impl PyNodeId {
    #[new]
    fn new(value: u64) -> Self {
        Self {
            inner: NodeId::new(value),
        }
    }

    #[getter]
    fn value(&self) -> u64 {
        self.inner.value()
    }

    fn __repr__(&self) -> String {
        format!("NodeId({})", self.inner.value())
    }

    fn __hash__(&self) -> u64 {
        self.inner.value()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __int__(&self) -> u64 {
        self.inner.value()
    }
}

/// A node in the build graph.
#[pyclass(name = "GraphNode")]
#[derive(Clone)]
pub struct PyGraphNode {
    pub(crate) inner: GraphNode,
}

#[pymethods]
impl PyGraphNode {
    #[new]
    fn new(id: PyNodeId, name: String, config: PyModelConfig) -> Self {
        Self {
            inner: GraphNode {
                id: id.inner,
                name,
                config: config.inner,
                metadata: HashMap::new(),
            },
        }
    }

    #[getter]
    fn id(&self) -> PyNodeId {
        PyNodeId {
            inner: self.inner.id,
        }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn config(&self) -> PyModelConfig {
        PyModelConfig {
            inner: self.inner.config.clone(),
        }
    }

    /// Get metadata as a Python dict.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        pythonize::pythonize(py, &self.inner.metadata)
            .map(|bound| bound.unbind())
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to convert metadata: {}",
                    e
                ))
            })
    }

    /// Add metadata and return self (builder pattern).
    fn with_metadata(&mut self, key: String, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let json_value: serde_json::Value = pythonize::depythonize(value).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to convert value: {}", e))
        })?;
        self.inner.metadata.insert(key, json_value);
        Ok(self.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "GraphNode(id={}, name='{}', config={})",
            self.inner.id.value(),
            self.inner.name,
            self.inner.config.name
        )
    }
}

/// Computation graph with nodes and edges.
#[pyclass(name = "BuildGraph")]
#[derive(Clone, Default)]
pub struct PyBuildGraph {
    pub(crate) inner: BuildGraph,
}

#[pymethods]
impl PyBuildGraph {
    #[new]
    fn new() -> Self {
        Self {
            inner: BuildGraph::default(),
        }
    }

    #[getter]
    fn nodes(&self) -> Vec<PyGraphNode> {
        self.inner
            .nodes
            .iter()
            .map(|n| PyGraphNode { inner: n.clone() })
            .collect()
    }

    #[getter]
    fn edges(&self) -> Vec<(PyNodeId, PyNodeId)> {
        self.inner
            .edges
            .iter()
            .map(|(from, to)| (PyNodeId { inner: *from }, PyNodeId { inner: *to }))
            .collect()
    }

    fn add_node(&mut self, name: &str, config: &PyModelConfig) -> PyNodeId {
        let id = self.inner.add_node(name, config.inner.clone());
        PyNodeId { inner: id }
    }

    fn add_edge(&mut self, from_id: PyNodeId, to_id: PyNodeId) {
        self.inner.add_edge(from_id.inner, to_id.inner);
    }

    fn node_count(&self) -> usize {
        self.inner.nodes.len()
    }

    fn edge_count(&self) -> usize {
        self.inner.edges.len()
    }

    fn stable_hash(&self) -> String {
        self.inner.stable_hash()
    }

    fn __repr__(&self) -> String {
        format!(
            "BuildGraph(nodes={}, edges={}, hash='{}')",
            self.inner.nodes.len(),
            self.inner.edges.len(),
            &self.inner.stable_hash()[..8]
        )
    }

    fn __len__(&self) -> usize {
        self.inner.nodes.len()
    }

    fn __bool__(&self) -> bool {
        !self.inner.nodes.is_empty()
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNodeId>()?;
    m.add_class::<PyGraphNode>()?;
    m.add_class::<PyBuildGraph>()?;
    Ok(())
}
