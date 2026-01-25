//! Error types and result aliases for Aphelion operations.
//!
//! This module provides a unified error type (`AphelionError`) for all Aphelion operations,
//! enabling consistent error handling across the framework. Errors are categorized by type
//! to allow fine-grained error recovery and reporting. The module also provides rich context
//! support for detailed error reporting and error chaining.

use std::fmt;
use std::io;
use thiserror::Error;

/// Context information for enhanced error reporting.
///
/// `ErrorContext` provides optional metadata about where and how an error occurred,
/// including the stage of processing, the specific field affected, and additional details.
/// This information is useful for debugging and user-facing error messages.
#[derive(Debug, Clone, Default)]
pub struct ErrorContext {
    /// The processing stage where the error occurred (e.g., "config_validation", "graph_build")
    pub stage: Option<String>,
    /// The specific field or component affected by the error
    pub field: Option<String>,
    /// Additional details or context about the error
    pub details: Option<String>,
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = [
            self.stage.as_ref().map(|s| format!("stage: {}", s)),
            self.field.as_ref().map(|s| format!("field: {}", s)),
            self.details.as_ref().map(|s| format!("details: {}", s)),
        ]
        .into_iter()
        .flatten()
        .collect();

        if !parts.is_empty() {
            write!(f, " ({})", parts.join(", "))
        } else {
            Ok(())
        }
    }
}

/// Unified error type for all Aphelion operations.
///
/// `AphelionError` encompasses all error conditions that can occur in the framework,
/// from configuration issues to backend failures. Each variant can carry context
/// information and support error chaining.
#[derive(Debug, Error)]
pub enum AphelionError {
    /// Configuration validation or parsing failed
    #[error("invalid configuration: {message}{}", .context)]
    InvalidConfig {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: ErrorContext,
    },
    /// Backend initialization, availability, or operation failed
    #[error("backend error: {message}{}", .context)]
    Backend {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: ErrorContext,
    },
    /// Model building or pipeline execution failed
    #[error("build error: {message}{}", .context)]
    Build {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: ErrorContext,
    },
    /// Data validation failed
    #[error("validation error: {message}{}", .context)]
    Validation {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: ErrorContext,
    },
    /// JSON serialization or deserialization failed
    #[error("serialization error: {message}{}", .context)]
    Serialization {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: ErrorContext,
    },
    /// File system or I/O operation failed
    #[error("io error: {message}{}", .context)]
    Io {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: ErrorContext,
    },
    /// Graph construction or validation failed
    #[error("graph error: {message}{}", .context)]
    Graph {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        context: ErrorContext,
    },
}

impl From<io::Error> for AphelionError {
    fn from(err: io::Error) -> Self {
        AphelionError::Io {
            message: err.to_string(),
            source: None,
            context: ErrorContext::default(),
        }
    }
}

/// Result type alias for Aphelion operations.
pub type AphelionResult<T> = Result<T, AphelionError>;

impl AphelionError {
    /// Creates a configuration error with a custom message.
    pub fn config_error(message: impl Into<String>) -> Self {
        AphelionError::InvalidConfig {
            message: message.into(),
            source: None,
            context: ErrorContext::default(),
        }
    }

    /// Creates a validation error with field context.
    pub fn validation_error(field: &str, message: impl Into<String>) -> Self {
        AphelionError::Validation {
            message: message.into(),
            source: None,
            context: ErrorContext {
                stage: None,
                field: Some(field.to_string()),
                details: None,
            },
        }
    }

    /// Creates a validation error with a custom message.
    pub fn validation(msg: impl Into<String>) -> Self {
        AphelionError::Validation {
            message: msg.into(),
            source: None,
            context: ErrorContext::default(),
        }
    }

    /// Creates a serialization error with a custom message.
    pub fn serialization(msg: impl Into<String>) -> Self {
        AphelionError::Serialization {
            message: msg.into(),
            source: None,
            context: ErrorContext::default(),
        }
    }

    /// Creates an I/O error with a custom message.
    pub fn io(msg: impl Into<String>) -> Self {
        AphelionError::Io {
            message: msg.into(),
            source: None,
            context: ErrorContext::default(),
        }
    }

    /// Creates a graph error with a custom message.
    pub fn graph(msg: impl Into<String>) -> Self {
        AphelionError::Graph {
            message: msg.into(),
            source: None,
            context: ErrorContext::default(),
        }
    }

    /// Creates a backend error with a custom message.
    pub fn backend(msg: impl Into<String>) -> Self {
        AphelionError::Backend {
            message: msg.into(),
            source: None,
            context: ErrorContext::default(),
        }
    }

    /// Creates a build error with a custom message.
    pub fn build(msg: impl Into<String>) -> Self {
        AphelionError::Build {
            message: msg.into(),
            source: None,
            context: ErrorContext::default(),
        }
    }

    /// Adds context specifying the processing stage.
    pub fn in_stage(mut self, stage: &str) -> Self {
        match &mut self {
            AphelionError::InvalidConfig { context, .. }
            | AphelionError::Backend { context, .. }
            | AphelionError::Build { context, .. }
            | AphelionError::Validation { context, .. }
            | AphelionError::Serialization { context, .. }
            | AphelionError::Io { context, .. }
            | AphelionError::Graph { context, .. } => {
                context.stage = Some(stage.to_string());
            }
        }
        self
    }

    /// Adds context specifying the affected field.
    pub fn for_field(mut self, field: &str) -> Self {
        match &mut self {
            AphelionError::InvalidConfig { context, .. }
            | AphelionError::Backend { context, .. }
            | AphelionError::Build { context, .. }
            | AphelionError::Validation { context, .. }
            | AphelionError::Serialization { context, .. }
            | AphelionError::Io { context, .. }
            | AphelionError::Graph { context, .. } => {
                context.field = Some(field.to_string());
            }
        }
        self
    }

    /// Adds context with detailed information.
    pub fn with_context(mut self, ctx: ErrorContext) -> Self {
        match &mut self {
            AphelionError::InvalidConfig { context, .. }
            | AphelionError::Backend { context, .. }
            | AphelionError::Build { context, .. }
            | AphelionError::Validation { context, .. }
            | AphelionError::Serialization { context, .. }
            | AphelionError::Io { context, .. }
            | AphelionError::Graph { context, .. } => {
                *context = ctx;
            }
        }
        self
    }

    /// Extracts the error message string.
    pub fn message(&self) -> &str {
        match self {
            AphelionError::InvalidConfig { message, .. } => message,
            AphelionError::Backend { message, .. } => message,
            AphelionError::Build { message, .. } => message,
            AphelionError::Validation { message, .. } => message,
            AphelionError::Serialization { message, .. } => message,
            AphelionError::Io { message, .. } => message,
            AphelionError::Graph { message, .. } => message,
        }
    }

    /// Extracts the optional context information.
    pub fn context(&self) -> &ErrorContext {
        match self {
            AphelionError::InvalidConfig { context, .. } => context,
            AphelionError::Backend { context, .. } => context,
            AphelionError::Build { context, .. } => context,
            AphelionError::Validation { context, .. } => context,
            AphelionError::Serialization { context, .. } => context,
            AphelionError::Io { context, .. } => context,
            AphelionError::Graph { context, .. } => context,
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
        let validation_err = AphelionError::validation("invalid input");
        assert_eq!(validation_err.to_string(), "validation error: invalid input");

        let serialization_err = AphelionError::serialization("bad format");
        assert_eq!(serialization_err.to_string(), "serialization error: bad format");

        let io_err = AphelionError::io("permission denied");
        assert_eq!(io_err.to_string(), "io error: permission denied");

        let graph_err = AphelionError::graph("invalid node");
        assert_eq!(graph_err.to_string(), "graph error: invalid node");
    }

    #[test]
    fn test_original_error_variants_still_work() {
        let config_err = AphelionError::config_error("missing key");
        assert_eq!(config_err.message(), "missing key");
        assert_eq!(config_err.to_string(), "invalid configuration: missing key");

        let backend_err = AphelionError::backend("connection failed");
        assert_eq!(backend_err.message(), "connection failed");
        assert_eq!(backend_err.to_string(), "backend error: connection failed");

        let build_err = AphelionError::build("compilation failed");
        assert_eq!(build_err.message(), "compilation failed");
        assert_eq!(build_err.to_string(), "build error: compilation failed");
    }

    #[test]
    fn test_result_type_with_validation() {
        let result: AphelionResult<i32> = Err(AphelionError::validation("number too large"));
        assert!(result.is_err());
        match result {
            Err(AphelionError::Validation { message, .. }) => {
                assert_eq!(message, "number too large");
            }
            _ => panic!("Expected Validation error"),
        }
    }

    #[test]
    fn test_result_type_with_serialization() {
        let result: AphelionResult<String> = Err(AphelionError::serialization("invalid utf8"));
        assert!(result.is_err());
        match result {
            Err(AphelionError::Serialization { message, .. }) => {
                assert_eq!(message, "invalid utf8");
            }
            _ => panic!("Expected Serialization error"),
        }
    }

    #[test]
    fn test_error_with_context() {
        let error = AphelionError::validation_error("hidden_size", "must be >= 64")
            .in_stage("config_validation");

        assert_eq!(error.message(), "must be >= 64");
        let ctx = error.context();
        assert_eq!(ctx.field.as_deref(), Some("hidden_size"));
        assert_eq!(ctx.stage.as_deref(), Some("config_validation"));
    }

    #[test]
    fn test_error_chaining() {
        let error = AphelionError::config_error("invalid configuration")
            .in_stage("config_parsing")
            .for_field("model_name");

        let ctx = error.context();
        assert_eq!(ctx.stage.as_deref(), Some("config_parsing"));
        assert_eq!(ctx.field.as_deref(), Some("model_name"));
        assert_eq!(error.message(), "invalid configuration");
    }

    #[test]
    fn test_error_display_includes_context() {
        let error = AphelionError::validation_error("batch_size", "must be positive")
            .in_stage("training_setup");

        let display_string = error.to_string();
        assert!(display_string.contains("validation error"));
        assert!(display_string.contains("must be positive"));
        assert!(display_string.contains("stage: training_setup"));
        assert!(display_string.contains("field: batch_size"));
    }

    #[test]
    fn test_error_context_struct() {
        let ctx = ErrorContext {
            stage: Some("build".to_string()),
            field: Some("param".to_string()),
            details: Some("critical".to_string()),
        };

        let error = AphelionError::build("failed").with_context(ctx);
        let retrieved_ctx = error.context();
        assert_eq!(retrieved_ctx.stage.as_deref(), Some("build"));
        assert_eq!(retrieved_ctx.field.as_deref(), Some("param"));
        assert_eq!(retrieved_ctx.details.as_deref(), Some("critical"));
    }

    #[test]
    fn test_helper_constructors() {
        let config_err = AphelionError::config_error("test");
        assert_eq!(config_err.message(), "test");

        let validation_err = AphelionError::validation_error("field", "message");
        assert_eq!(validation_err.message(), "message");
        assert_eq!(validation_err.context().field.as_deref(), Some("field"));

        let backend_err = AphelionError::backend("unavailable");
        assert_eq!(backend_err.message(), "unavailable");
    }

    #[test]
    fn test_error_context_display() {
        let ctx = ErrorContext {
            stage: Some("validation".to_string()),
            field: Some("input".to_string()),
            details: None,
        };

        let display = ctx.to_string();
        assert!(display.contains("stage: validation"));
        assert!(display.contains("field: input"));

        let empty_ctx = ErrorContext::default();
        assert_eq!(empty_ctx.to_string(), "");
    }
}
