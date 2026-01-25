use thiserror::Error;

#[derive(Debug, Error)]
pub enum AphelionError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("build error: {0}")]
    Build(String),
}

pub type AphelionResult<T> = Result<T, AphelionError>;
