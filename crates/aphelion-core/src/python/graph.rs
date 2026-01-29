//! Python bindings for graph types.
//!
//! Provides DAG-based model architecture representation with deterministic hashing.
//! Graphs consist of nodes (model components) and edges (data flow dependencies).

use pyo3::prelude::*;
use std::collections::HashMap;

use crate::graph::{BuildGraph, GraphNode, NodeId};

use super::config::PyModelConfig;

/// Unique identifier for a graph node.
///
/// NodeIds are assigned sequentially when nodes are added to a graph. They
/// are hashable and can be used as dictionary keys.
///
/// Example:
///     >>> graph = BuildGraph()
///     >>> node_id = graph.add_node("encoder", config)
///     >>> int(node_id)
///     0
#[pyclass(name = "NodeId")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyNodeId {
    pub(crate) inner: NodeId,
}

#[pymethods]
impl PyNodeId {
    /// Create a NodeId with a specific value.
    ///
    /// Typically you get NodeIds from BuildGraph.add_node() rather than
    /// creating them directly.
    ///
    /// Args:
    ///     value: Numeric identifier.
    #[new]
    #[pyo3(text_signature = "(value)")]
    fn new(value: u64) -> Self {
        Self {
            inner: NodeId::new(value),
        }
    }

    /// Numeric value of this identifier.
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

/// A node in the build graph representing a model component.
///
/// Each node contains:
/// - A unique identifier
/// - A human-readable name
/// - A ModelConfig with parameters
/// - Optional metadata for custom annotations
///
/// Example:
///     >>> node = GraphNode(NodeId(0), "encoder", config)
///     >>> node.name
///     'encoder'
///     >>> node = node.with_metadata("layer_type", "attention")
#[pyclass(name = "GraphNode")]
#[derive(Clone)]
pub struct PyGraphNode {
    pub(crate) inner: GraphNode,
}

#[pymethods]
impl PyGraphNode {
    /// Create a graph node.
    ///
    /// Args:
    ///     id: Unique node identifier.
    ///     name: Human-readable name for this component.
    ///     config: ModelConfig with component parameters.
    #[new]
    #[pyo3(text_signature = "(id, name, config)")]
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

    /// Node identifier.
    #[getter]
    fn id(&self) -> PyNodeId {
        PyNodeId {
            inner: self.inner.id,
        }
    }

    /// Human-readable component name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Model configuration for this component.
    #[getter]
    fn config(&self) -> PyModelConfig {
        PyModelConfig {
            inner: self.inner.config.clone(),
        }
    }

    /// Custom metadata as a dictionary.
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        pythonize::pythonize(py, &self.inner.metadata)
            .map(|bound| bound.unbind())
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Failed to convert metadata: {}",
                    e
                ))
            })
    }

    /// Add metadata to this node.
    ///
    /// Metadata can store arbitrary annotations like optimization hints,
    /// layer types, or debugging information.
    ///
    /// Args:
    ///     key: Metadata key.
    ///     value: Any JSON-serializable value.
    ///
    /// Returns:
    ///     Self with metadata added.
    #[pyo3(text_signature = "(key, value)")]
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

/// Directed acyclic graph representing model architecture.
///
/// The graph stores nodes (model components) and directed edges (data flow).
/// It provides deterministic hashing: identical graphs produce identical
/// SHA-256 hashes regardless of construction order.
///
/// Why deterministic hashing matters:
/// - Reproducible builds: same config = same hash = same model
/// - Cache validation: detect when model architecture changed
/// - Audit trails: cryptographic proof of model configuration
///
/// Example:
///     >>> graph = BuildGraph()
///     >>> encoder = graph.add_node("encoder", enc_config)
///     >>> decoder = graph.add_node("decoder", dec_config)
///     >>> graph.add_edge(encoder, decoder)
///     >>> graph.stable_hash()
///     'a1b2c3d4e5f6...'
///
/// Attributes:
///     nodes (list): All GraphNode objects in the graph.
///     edges (list): All edges as (from_id, to_id) tuples.
#[pyclass(name = "BuildGraph")]
#[derive(Clone, Default)]
pub struct PyBuildGraph {
    pub(crate) inner: BuildGraph,
}

#[pymethods]
impl PyBuildGraph {
    /// Create an empty build graph.
    #[new]
    fn new() -> Self {
        Self {
            inner: BuildGraph::default(),
        }
    }

    /// All nodes in the graph.
    #[getter]
    fn nodes(&self) -> Vec<PyGraphNode> {
        self.inner
            .nodes
            .iter()
            .map(|n| PyGraphNode { inner: n.clone() })
            .collect()
    }

    /// All edges as (source, target) pairs.
    #[getter]
    fn edges(&self) -> Vec<(PyNodeId, PyNodeId)> {
        self.inner
            .edges
            .iter()
            .map(|(from, to)| (PyNodeId { inner: *from }, PyNodeId { inner: *to }))
            .collect()
    }

    /// Add a node to the graph.
    ///
    /// Args:
    ///     name: Human-readable component name.
    ///     config: ModelConfig for this component.
    ///
    /// Returns:
    ///     NodeId for the new node, usable in add_edge().
    ///
    /// Example:
    ///     >>> encoder_id = graph.add_node("encoder", encoder_config)
    #[pyo3(text_signature = "(name, config)")]
    fn add_node(&mut self, name: &str, config: &PyModelConfig) -> PyNodeId {
        let id = self.inner.add_node(name, config.inner.clone());
        PyNodeId { inner: id }
    }

    /// Add a directed edge between nodes.
    ///
    /// Edges represent data flow: data flows from source to target.
    /// The graph must remain acyclic.
    ///
    /// Args:
    ///     from_id: Source node (data producer).
    ///     to_id: Target node (data consumer).
    ///
    /// Example:
    ///     >>> graph.add_edge(encoder_id, decoder_id)
    #[pyo3(text_signature = "(from_id, to_id)")]
    fn add_edge(&mut self, from_id: PyNodeId, to_id: PyNodeId) {
        self.inner.add_edge(from_id.inner, to_id.inner);
    }

    /// Number of nodes in the graph.
    fn node_count(&self) -> usize {
        self.inner.nodes.len()
    }

    /// Number of edges in the graph.
    fn edge_count(&self) -> usize {
        self.inner.edges.len()
    }

    /// Compute deterministic SHA-256 hash of the graph.
    ///
    /// The hash is computed over canonicalized node and edge data.
    /// Identical graphs produce identical hashes regardless of
    /// construction order.
    ///
    /// Returns:
    ///     64-character hexadecimal hash string.
    ///
    /// Example:
    ///     >>> graph.stable_hash()
    ///     'a1b2c3d4e5f6789...'
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
