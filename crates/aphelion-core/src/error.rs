//! Error types and result aliases for Aphelion operations.
//!
//! This module provides a unified error type (`AphelionError`) for all Aphelion operations,
//! enabling consistent error handling across the framework. Errors are categorized by type
//! to allow fine-grained error recovery and reporting.

use std::io;
use thiserror::Error;

/// Unified error type for all Aphelion operations.
///
/// `AphelionError` encompasses all error conditions that can occur in the framework,
/// from configuration issues to backend failures. Each variant is designed to provide
/// clear error messages and categorization for debugging and user feedback.
///
/// # Variants
///
/// - `InvalidConfig(String)` - Configuration validation or parsing failed
/// - `Backend(String)` - Backend initialization, availability, or operation failed
/// - `Build(String)` - Model building or pipeline execution failed
/// - `Validation(String)` - Data validation failed
/// - `Serialization(String)` - JSON serialization or deserialization failed
/// - `Io(String)` - File system or I/O operation failed
/// - `Graph(String)` - Graph construction or validation failed (e.g., cycles detected)
///
/// # Examples
///
/// ```
/// use aphelion_core::error::{AphelionError, AphelionResult};
///
/// // Creating specific errors
/// let config_error = AphelionError::InvalidConfig("missing field 'name'".to_string());
/// let validation_error = AphelionError::validation("model version must be >= 1.0.0");
///
/// // Using in Result types
/// fn validate_version(v: &str) -> AphelionResult<()> {
///     if v.starts_with("v") {
///         Err(AphelionError::InvalidConfig("version must not start with 'v'".to_string()))
///     } else {
///         Ok(())
///     }
/// }
/// ```
#[derive(Debug, Error)]
pub enum AphelionError {
    /// Configuration validation or parsing failed
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// Backend initialization, availability, or operation failed
    #[error("backend error: {0}")]
    Backend(String),
    /// Model building or pipeline execution failed
    #[error("build error: {0}")]
    Build(String),
    /// Data validation failed
    #[error("validation error: {0}")]
    Validation(String),
    /// JSON serialization or deserialization failed
    #[error("serialization error: {0}")]
    Serialization(String),
    /// File system or I/O operation failed
    #[error("io error: {0}")]
    Io(String),
    /// Graph construction or validation failed
    #[error("graph error: {0}")]
    Graph(String),
}

impl From<io::Error> for AphelionError {
    fn from(err: io::Error) -> Self {
        AphelionError::Io(err.to_string())
    }
}

/// Result type alias for Aphelion operations.
///
/// All Aphelion operations that can fail return `AphelionResult<T>`, which is
/// equivalent to `Result<T, AphelionError>`. This provides a consistent error
/// handling interface throughout the framework.
///
/// # Examples
///
/// ```ignore
/// use aphelion_core::error::AphelionResult;
///
/// fn build_model() -> AphelionResult<BuildGraph> {
///     // ... implementation ...
/// }
/// ```
pub type AphelionResult<T> = Result<T, AphelionError>;

impl AphelionError {
    /// Create a validation error with a custom message
    pub fn validation(msg: impl Into<String>) -> Self {
        AphelionError::Validation(msg.into())
    }

    /// Create a serialization error with a custom message
    pub fn serialization(msg: impl Into<String>) -> Self {
        AphelionError::Serialization(msg.into())
    }

    /// Create an IO error with a custom message
    pub fn io(msg: impl Into<String>) -> Self {
        AphelionError::Io(msg.into())
    }

    /// Create a graph error with a custom message
    pub fn graph(msg: impl Into<String>) -> Self {
        AphelionError::Graph(msg.into())
    }

    /// Get a string reference to the error message
    pub fn message(&self) -> &str {
        match self {
            AphelionError::InvalidConfig(msg) => msg,
            AphelionError::Backend(msg) => msg,
            AphelionError::Build(msg) => msg,
            AphelionError::Validation(msg) => msg,
            AphelionError::Serialization(msg) => msg,
            AphelionError::Io(msg) => msg,
            AphelionError::Graph(msg) => msg,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_error_creation() {
        let err = AphelionError::validation("field 'name' is required");
        assert_eq!(err.message(), "field 'name' is required");
    }

    #[test]
    fn test_serialization_error_creation() {
        let err = AphelionError::serialization("failed to serialize JSON");
        assert_eq!(err.message(), "failed to serialize JSON");
    }

    #[test]
    fn test_io_error_creation() {
        let err = AphelionError::io("file not found");
        assert_eq!(err.message(), "file not found");
    }

    #[test]
    fn test_graph_error_creation() {
        let err = AphelionError::graph("cycle detected in graph");
        assert_eq!(err.message(), "cycle detected in graph");
    }

    #[test]
    fn test_io_error_from_std() {
        let std_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let aphelion_err: AphelionError = std_err.into();
        assert!(aphelion_err.message().contains("file not found"));
    }

    #[test]
    fn test_error_display_messages() {
        let validation_err = AphelionError::Validation("invalid input".to_string());
        assert_eq!(validation_err.to_string(), "validation error: invalid input");

        let serialization_err = AphelionError::Serialization("bad format".to_string());
        assert_eq!(serialization_err.to_string(), "serialization error: bad format");

        let io_err = AphelionError::Io("permission denied".to_string());
        assert_eq!(io_err.to_string(), "io error: permission denied");

        let graph_err = AphelionError::Graph("invalid node".to_string());
        assert_eq!(graph_err.to_string(), "graph error: invalid node");
    }

    #[test]
    fn test_original_error_variants_still_work() {
        let config_err = AphelionError::InvalidConfig("missing key".to_string());
        assert_eq!(config_err.message(), "missing key");
        assert_eq!(config_err.to_string(), "invalid configuration: missing key");

        let backend_err = AphelionError::Backend("connection failed".to_string());
        assert_eq!(backend_err.message(), "connection failed");
        assert_eq!(backend_err.to_string(), "backend error: connection failed");

        let build_err = AphelionError::Build("compilation failed".to_string());
        assert_eq!(build_err.message(), "compilation failed");
        assert_eq!(build_err.to_string(), "build error: compilation failed");
    }

    #[test]
    fn test_result_type_with_validation() {
        let result: AphelionResult<i32> = Err(AphelionError::validation("number too large"));
        assert!(result.is_err());
        match result {
            Err(AphelionError::Validation(msg)) => {
                assert_eq!(msg, "number too large");
            }
            _ => panic!("Expected Validation error"),
        }
    }

    #[test]
    fn test_result_type_with_serialization() {
        let result: AphelionResult<String> = Err(AphelionError::serialization("invalid utf8"));
        assert!(result.is_err());
        match result {
            Err(AphelionError::Serialization(msg)) => {
                assert_eq!(msg, "invalid utf8");
            }
            _ => panic!("Expected Serialization error"),
        }
    }
}
