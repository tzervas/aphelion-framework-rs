//! WASM bindings for graph types.

use super::config::JsModelConfig;
use crate::graph::{BuildGraph, GraphNode, NodeId};
use wasm_bindgen::prelude::*;

/// Unique identifier for a graph node.
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct JsNodeId {
    inner: NodeId,
}

#[wasm_bindgen]
impl JsNodeId {
    /// Get the numeric value of this node ID.
    #[wasm_bindgen(getter)]
    pub fn value(&self) -> u64 {
        self.inner.value()
    }

    /// Create a string representation.
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string_js(&self) -> String {
        format!("NodeId({})", self.inner.value())
    }
}

impl JsNodeId {
    pub(crate) fn from_inner(inner: NodeId) -> Self {
        Self { inner }
    }

    pub(crate) fn inner(&self) -> NodeId {
        self.inner
    }
}

/// A node in the build graph representing a model component.
#[wasm_bindgen]
pub struct JsGraphNode {
    inner: GraphNode,
}

#[wasm_bindgen]
impl JsGraphNode {
    /// Get the node's unique identifier.
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> JsNodeId {
        JsNodeId::from_inner(self.inner.id)
    }

    /// Get the node's name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Get the node's configuration.
    #[wasm_bindgen(getter)]
    pub fn config(&self) -> JsModelConfig {
        JsModelConfig::from_inner(self.inner.config.clone())
    }

    /// Get the node's metadata as a JSON object.
    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> Result<JsValue, JsError> {
        serde_wasm_bindgen::to_value(&self.inner.metadata)
            .map_err(|e| JsError::new(&format!("Failed to serialize metadata: {}", e)))
    }
}

impl JsGraphNode {
    pub(crate) fn from_inner(inner: GraphNode) -> Self {
        Self { inner }
    }
}

/// Directed acyclic graph for model architecture.
///
/// BuildGraph represents the structure of a model as nodes (components)
/// and edges (data flow). The graph produces a deterministic SHA-256 hash
/// for reproducibility.
#[wasm_bindgen]
pub struct JsBuildGraph {
    inner: BuildGraph,
}

#[wasm_bindgen]
impl JsBuildGraph {
    /// Create an empty build graph.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: BuildGraph::default(),
        }
    }

    /// Add a node to the graph.
    #[wasm_bindgen(js_name = addNode)]
    pub fn add_node(&mut self, name: &str, config: &JsModelConfig) -> JsNodeId {
        let node_id = self
            .inner
            .add_node(name.to_string(), config.inner().clone());
        JsNodeId::from_inner(node_id)
    }

    /// Add an edge between two nodes.
    #[wasm_bindgen(js_name = addEdge)]
    pub fn add_edge(&mut self, from: &JsNodeId, to: &JsNodeId) {
        self.inner.add_edge(from.inner(), to.inner());
    }

    /// Get the number of nodes in the graph.
    #[wasm_bindgen(js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Get the number of edges in the graph.
    #[wasm_bindgen(js_name = edgeCount)]
    pub fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Check if the graph contains a cycle.
    #[wasm_bindgen(js_name = hasCycle)]
    pub fn has_cycle(&self) -> bool {
        self.inner.has_cycle()
    }

    /// Get the deterministic SHA-256 hash of the graph.
    #[wasm_bindgen(js_name = stableHash)]
    pub fn stable_hash(&self) -> String {
        self.inner.stable_hash()
    }

    /// Get a topological ordering of node IDs.
    #[wasm_bindgen(js_name = topologicalSort)]
    pub fn topological_sort(&self) -> Result<Vec<u64>, JsError> {
        self.inner
            .topological_sort()
            .map(|ids| ids.into_iter().map(|id| id.value()).collect())
            .map_err(|e| JsError::new(&format!("Topological sort failed: {}", e)))
    }

    /// Export the graph in DOT format for visualization.
    #[wasm_bindgen(js_name = toDot)]
    pub fn to_dot(&self) -> String {
        self.inner.to_dot()
    }

    /// Serialize the graph to JSON.
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> Result<String, JsError> {
        serde_json::to_string(&self.inner)
            .map_err(|e| JsError::new(&format!("Failed to serialize: {}", e)))
    }

    /// Deserialize a graph from JSON.
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(json: &str) -> Result<JsBuildGraph, JsError> {
        let inner: BuildGraph = serde_json::from_str(json)
            .map_err(|e| JsError::new(&format!("Failed to deserialize: {}", e)))?;
        Ok(Self { inner })
    }

    /// Get all nodes as an array.
    #[wasm_bindgen(js_name = getNodes)]
    pub fn get_nodes(&self) -> Vec<JsGraphNode> {
        self.inner
            .nodes
            .iter()
            .map(|n| JsGraphNode::from_inner(n.clone()))
            .collect()
    }
}

impl Default for JsBuildGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl JsBuildGraph {
    pub(crate) fn into_inner(self) -> BuildGraph {
        self.inner
    }

    pub(crate) fn from_inner(inner: BuildGraph) -> Self {
        Self { inner }
    }
}
