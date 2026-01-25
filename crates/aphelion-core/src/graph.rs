//! Graph construction and manipulation for model architectures.
//!
//! This module provides types and algorithms for building directed acyclic graphs (DAGs)
//! representing model architectures. Graphs support cycle detection, topological sorting,
//! stable hashing, and DOT format export for visualization.

use crate::config::ModelConfig;
use crate::error::AphelionError;
use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Unique identifier for a graph node.
///
/// `NodeId` is a thin wrapper around `u64` that provides type safety and clarity.
/// It implements `Copy`, `Hash`, and `Ord` to enable efficient use in collections
/// and graph algorithms.
///
/// # Examples
///
/// ```
/// use aphelion_core::graph::NodeId;
///
/// let id1 = NodeId::new(1);
/// let id2 = NodeId::new(2);
/// assert_eq!(id1.value(), 1);
/// assert_ne!(id1, id2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeId(u64);

impl NodeId {
    /// Creates a new `NodeId` with the given value.
    ///
    /// # Arguments
    ///
    /// * `value` - The unique identifier value
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::graph::NodeId;
    ///
    /// let id = NodeId::new(42);
    /// assert_eq!(id.value(), 42);
    /// ```
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    /// Extracts the numeric value from this `NodeId`.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::graph::NodeId;
    ///
    /// let id = NodeId::new(123);
    /// assert_eq!(id.value(), 123);
    /// ```
    pub fn value(self) -> u64 {
        self.0
    }
}

/// A node in the computation graph.
///
/// `GraphNode` represents a single layer, operation, or model component in a computation graph.
/// Each node contains a unique identifier, name, configuration, and optional metadata.
///
/// # Fields
///
/// * `id` - Unique node identifier
/// * `name` - Human-readable node name (e.g., "embedding_layer", "attention")
/// * `config` - Model configuration for this node
/// * `metadata` - Additional metadata stored as JSON values
///
/// # Examples
///
/// ```
/// use aphelion_core::graph::{GraphNode, NodeId};
/// use aphelion_core::config::ModelConfig;
///
/// let config = ModelConfig::new("layer", "1.0.0");
/// let node = GraphNode {
///     id: NodeId::new(1),
///     name: "embedding".to_string(),
///     config,
///     metadata: std::collections::HashMap::new(),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique node identifier
    pub id: NodeId,
    /// Human-readable node name
    pub name: String,
    /// Model configuration for this node
    pub config: ModelConfig,
    /// Additional metadata as JSON values
    pub metadata: HashMap<String, serde_json::Value>,
}

impl GraphNode {
    /// Adds or updates metadata for this node.
    ///
    /// Uses the builder pattern to allow chaining multiple metadata assignments.
    ///
    /// # Arguments
    ///
    /// * `key` - Metadata key
    /// * `value` - Metadata value as a JSON value
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::graph::{GraphNode, NodeId};
    /// use aphelion_core::config::ModelConfig;
    ///
    /// let config = ModelConfig::new("layer", "1.0.0");
    /// let node = GraphNode {
    ///     id: NodeId::new(1),
    ///     name: "dense".to_string(),
    ///     config,
    ///     metadata: std::collections::HashMap::new(),
    /// }.with_metadata("units", serde_json::json!(256))
    ///  .with_metadata("activation", serde_json::json!("relu"));
    ///
    /// assert_eq!(node.metadata.len(), 2);
    /// ```
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// A directed acyclic graph (DAG) representing a model architecture.
///
/// `BuildGraph` stores the nodes and edges of a computation graph, enabling operations
/// such as cycle detection, topological sorting, hashing, and DOT format export.
/// All graphs are validated to be acyclic to support deterministic execution.
///
/// # Fields
///
/// * `nodes` - Vector of graph nodes
/// * `edges` - Vector of edges represented as (from_id, to_id) tuples
///
/// # Examples
///
/// ```
/// use aphelion_core::graph::BuildGraph;
/// use aphelion_core::config::ModelConfig;
///
/// let mut graph = BuildGraph::default();
/// let config = ModelConfig::new("model", "1.0.0");
///
/// let node1 = graph.add_node("input", config.clone());
/// let node2 = graph.add_node("hidden", config.clone());
/// let node3 = graph.add_node("output", config);
///
/// graph.add_edge(node1, node2);
/// graph.add_edge(node2, node3);
///
/// // Verify no cycles
/// assert!(!graph.has_cycle());
///
/// // Get topological order
/// let topo = graph.topological_sort().expect("valid DAG");
/// assert_eq!(topo.len(), 3);
/// ```
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BuildGraph {
    /// All nodes in the graph
    pub nodes: Vec<GraphNode>,
    /// All edges as (from, to) node ID pairs
    pub edges: Vec<(NodeId, NodeId)>,
}

impl BuildGraph {
    pub fn add_node(&mut self, name: impl Into<String>, config: ModelConfig) -> NodeId {
        let id = NodeId::new(self.nodes.len() as u64 + 1);
        self.nodes.push(GraphNode {
            id,
            name: name.into(),
            config,
            metadata: HashMap::new(),
        });
        id
    }

    pub fn add_edge(&mut self, from: NodeId, to: NodeId) {
        self.edges.push((from, to));
    }

    /// Deterministic hash for traceability and reproducibility.
    pub fn stable_hash(&self) -> String {
        let mut hasher = Sha256::new();
        for node in &self.nodes {
            hasher.update(node.id.value().to_le_bytes());
            hasher.update(node.name.as_bytes());
            hasher.update(serde_json::to_vec(&node.config).unwrap_or_default());
        }
        for (from, to) in &self.edges {
            hasher.update(from.value().to_le_bytes());
            hasher.update(to.value().to_le_bytes());
        }
        hex::encode(hasher.finalize())
    }

    /// Returns the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Detects if the graph contains any cycles using depth-first search.
    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for node in &self.nodes {
            if !visited.contains(&node.id) {
                if self.has_cycle_dfs(node.id, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }
        false
    }

    /// Depth-first search helper for cycle detection.
    ///
    /// Recursively visits nodes, maintaining a recursion stack to detect back edges
    /// that would indicate a cycle.
    ///
    /// # Arguments
    ///
    /// * `node_id` - Current node being visited
    /// * `visited` - Set of all visited nodes
    /// * `rec_stack` - Current recursion stack
    fn has_cycle_dfs(
        &self,
        node_id: NodeId,
        visited: &mut HashSet<NodeId>,
        rec_stack: &mut HashSet<NodeId>,
    ) -> bool {
        visited.insert(node_id);
        rec_stack.insert(node_id);

        // Find all outgoing edges from this node
        for (from, to) in &self.edges {
            if *from == node_id {
                if !visited.contains(to) {
                    if self.has_cycle_dfs(*to, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(to) {
                    return true;
                }
            }
        }

        rec_stack.remove(&node_id);
        false
    }

    /// Returns nodes in topological order using Kahn's algorithm.
    ///
    /// Topological ordering ensures that for every directed edge (u, v), node u
    /// comes before v in the ordering. This is essential for determining a valid
    /// execution order for the graph. Returns an error if the graph contains a cycle.
    ///
    /// # Errors
    ///
    /// Returns `AphelionError::Build` if the graph contains a cycle.
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::graph::BuildGraph;
    /// use aphelion_core::config::ModelConfig;
    ///
    /// let mut graph = BuildGraph::default();
    /// let config = ModelConfig::new("model", "1.0.0");
    ///
    /// let n1 = graph.add_node("n1", config.clone());
    /// let n2 = graph.add_node("n2", config.clone());
    /// let n3 = graph.add_node("n3", config);
    ///
    /// graph.add_edge(n1, n2);
    /// graph.add_edge(n2, n3);
    ///
    /// let sorted = graph.topological_sort().expect("valid DAG");
    /// assert_eq!(sorted.len(), 3);
    /// assert_eq!(sorted[0], n1);
    /// assert_eq!(sorted[2], n3);
    /// ```
    pub fn topological_sort(&self) -> Result<Vec<NodeId>, AphelionError> {
        if self.has_cycle() {
            return Err(AphelionError::build(
                "Cannot perform topological sort: graph contains a cycle",
            ));
        }

        // Calculate in-degrees
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        for node in &self.nodes {
            in_degree.insert(node.id, 0);
        }

        for (_, to) in &self.edges {
            *in_degree.entry(*to).or_insert(0) += 1;
        }

        // Find all nodes with in-degree 0
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        for node in &self.nodes {
            if in_degree[&node.id] == 0 {
                queue.push_back(node.id);
            }
        }

        let mut result = Vec::new();
        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);

            // For each outgoing edge from this node
            for (from, to) in &self.edges {
                if *from == node_id {
                    let degree = in_degree.entry(*to).or_insert(0);
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(*to);
                    }
                }
            }
        }

        if result.len() != self.nodes.len() {
            return Err(AphelionError::build(
                "Topological sort failed: not all nodes were processed",
            ));
        }

        Ok(result)
    }

    /// Exports the graph in DOT (GraphViz) format for visualization.
    ///
    /// The DOT format can be used with GraphViz tools to generate visual representations
    /// of the graph. This is useful for debugging and understanding model architecture.
    ///
    /// # Returns
    ///
    /// A string containing the DOT representation of the graph
    ///
    /// # Examples
    ///
    /// ```
    /// use aphelion_core::graph::BuildGraph;
    /// use aphelion_core::config::ModelConfig;
    ///
    /// let mut graph = BuildGraph::default();
    /// let config = ModelConfig::new("model", "1.0.0");
    ///
    /// let n1 = graph.add_node("input", config.clone());
    /// let n2 = graph.add_node("output", config);
    /// graph.add_edge(n1, n2);
    ///
    /// let dot = graph.to_dot();
    /// assert!(dot.contains("digraph BuildGraph"));
    /// assert!(dot.contains("input"));
    /// assert!(dot.contains("output"));
    /// assert!(dot.contains("->"));
    /// ```
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph BuildGraph {\n");

        // Add nodes
        for node in &self.nodes {
            dot.push_str(&format!("    \"{}\" [label=\"{}\"];\n", node.id.value(), node.name));
        }

        // Add edges
        for (from, to) in &self.edges {
            dot.push_str(&format!("    \"{}\" -> \"{}\";\n", from.value(), to.value()));
        }

        dot.push_str("}\n");
        dot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    #[test]
    fn test_node_count() {
        let mut graph = BuildGraph::default();
        assert_eq!(graph.node_count(), 0);

        let config = ModelConfig::new("model1", "1.0");
        graph.add_node("Node 1", config.clone());
        assert_eq!(graph.node_count(), 1);

        graph.add_node("Node 2", config);
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn test_edge_count() {
        let mut graph = BuildGraph::default();
        let config = ModelConfig::new("model", "1.0");

        let node1 = graph.add_node("Node 1", config.clone());
        let node2 = graph.add_node("Node 2", config.clone());
        let node3 = graph.add_node("Node 3", config);

        assert_eq!(graph.edge_count(), 0);

        graph.add_edge(node1, node2);
        assert_eq!(graph.edge_count(), 1);

        graph.add_edge(node2, node3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_no_cycle() {
        let mut graph = BuildGraph::default();
        let config = ModelConfig::new("model", "1.0");

        let node1 = graph.add_node("Node 1", config.clone());
        let node2 = graph.add_node("Node 2", config.clone());
        let node3 = graph.add_node("Node 3", config);

        graph.add_edge(node1, node2);
        graph.add_edge(node2, node3);

        assert!(!graph.has_cycle());
    }

    #[test]
    fn test_has_cycle() {
        let mut graph = BuildGraph::default();
        let config = ModelConfig::new("model", "1.0");

        let node1 = graph.add_node("Node 1", config.clone());
        let node2 = graph.add_node("Node 2", config.clone());
        let node3 = graph.add_node("Node 3", config);

        graph.add_edge(node1, node2);
        graph.add_edge(node2, node3);
        graph.add_edge(node3, node1); // Creates a cycle

        assert!(graph.has_cycle());
    }

    #[test]
    fn test_self_cycle() {
        let mut graph = BuildGraph::default();
        let config = ModelConfig::new("model", "1.0");

        let node1 = graph.add_node("Node 1", config);
        graph.add_edge(node1, node1); // Self-loop

        assert!(graph.has_cycle());
    }

    #[test]
    fn test_topological_sort_valid() {
        let mut graph = BuildGraph::default();
        let config = ModelConfig::new("model", "1.0");

        let node1 = graph.add_node("Node 1", config.clone());
        let node2 = graph.add_node("Node 2", config.clone());
        let node3 = graph.add_node("Node 3", config);

        graph.add_edge(node1, node2);
        graph.add_edge(node2, node3);

        let result = graph.topological_sort();
        assert!(result.is_ok());

        let sorted = result.unwrap();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0], node1);
        assert_eq!(sorted[1], node2);
        assert_eq!(sorted[2], node3);
    }

    #[test]
    fn test_topological_sort_with_cycle() {
        let mut graph = BuildGraph::default();
        let config = ModelConfig::new("model", "1.0");

        let node1 = graph.add_node("Node 1", config.clone());
        let node2 = graph.add_node("Node 2", config.clone());
        let node3 = graph.add_node("Node 3", config);

        graph.add_edge(node1, node2);
        graph.add_edge(node2, node3);
        graph.add_edge(node3, node1); // Creates a cycle

        let result = graph.topological_sort();
        assert!(result.is_err());
    }

    #[test]
    fn test_to_dot() {
        let mut graph = BuildGraph::default();
        let config = ModelConfig::new("model", "1.0");

        let node1 = graph.add_node("Node 1", config.clone());
        let node2 = graph.add_node("Node 2", config);

        graph.add_edge(node1, node2);

        let dot = graph.to_dot();
        assert!(dot.contains("digraph BuildGraph"));
        assert!(dot.contains("Node 1"));
        assert!(dot.contains("Node 2"));
        assert!(dot.contains("->"));
    }

    #[test]
    fn test_graph_node_metadata() {
        let config = ModelConfig::new("model", "1.0");
        let node = GraphNode {
            id: NodeId::new(1),
            name: "Test Node".to_string(),
            config,
            metadata: HashMap::new(),
        };

        let updated_node = node.with_metadata("key1", serde_json::json!("value1"));
        assert_eq!(updated_node.metadata.get("key1").unwrap(), "value1");

        let updated_node = updated_node.with_metadata("key2", serde_json::json!(42));
        assert_eq!(updated_node.metadata.get("key1").unwrap(), "value1");
        assert_eq!(updated_node.metadata.get("key2").unwrap(), 42);
    }

    #[test]
    fn test_graph_serialization() {
        let mut graph = BuildGraph::default();
        let config = ModelConfig::new("model", "1.0");

        let node1 = graph.add_node("Node 1", config.clone());
        let node2 = graph.add_node("Node 2", config);

        graph.add_edge(node1, node2);

        // Serialize to JSON
        let json = serde_json::to_string(&graph).unwrap();
        assert!(!json.is_empty());

        // Deserialize back
        let deserialized: BuildGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.node_count(), 2);
        assert_eq!(deserialized.edge_count(), 1);
    }

    #[test]
    fn test_complex_topological_sort() {
        let mut graph = BuildGraph::default();
        let config = ModelConfig::new("model", "1.0");

        // Create a more complex DAG: 1->2, 1->3, 2->4, 3->4
        let node1 = graph.add_node("Node 1", config.clone());
        let node2 = graph.add_node("Node 2", config.clone());
        let node3 = graph.add_node("Node 3", config.clone());
        let node4 = graph.add_node("Node 4", config);

        graph.add_edge(node1, node2);
        graph.add_edge(node1, node3);
        graph.add_edge(node2, node4);
        graph.add_edge(node3, node4);

        let result = graph.topological_sort();
        assert!(result.is_ok());

        let sorted = result.unwrap();
        assert_eq!(sorted.len(), 4);
        // node1 should come first
        assert_eq!(sorted[0], node1);
        // node4 should come last
        assert_eq!(sorted[3], node4);
    }
}
