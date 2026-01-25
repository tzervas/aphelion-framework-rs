use crate::config::ModelConfig;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(u64);

impl NodeId {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn value(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: NodeId,
    pub name: String,
    pub config: ModelConfig,
}

#[derive(Debug, Default, Clone)]
pub struct BuildGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<(NodeId, NodeId)>,
}

impl BuildGraph {
    pub fn add_node(&mut self, name: impl Into<String>, config: ModelConfig) -> NodeId {
        let id = NodeId::new(self.nodes.len() as u64 + 1);
        self.nodes.push(GraphNode {
            id,
            name: name.into(),
            config,
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
}
