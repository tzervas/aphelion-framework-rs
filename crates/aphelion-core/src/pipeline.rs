use crate::backend::{Backend, ModelBuilder};
use crate::config::ConfigSpec;
use crate::diagnostics::{TraceEvent, TraceSink};
use crate::error::{AphelionError, AphelionResult};
use crate::graph::BuildGraph;
use std::time::SystemTime;

/// Build context containing backend + trace sink.
pub struct BuildContext<'a> {
    pub backend: &'a dyn Backend,
    pub trace: &'a dyn TraceSink,
}

/// A minimal build pipeline that can be extended with stages.
pub struct BuildPipeline;

impl BuildPipeline {
    pub fn build<M: ModelBuilder<Output = BuildGraph>>(
        model: &M,
        ctx: BuildContext<'_>,
    ) -> AphelionResult<BuildGraph> {
        ctx.trace.record(TraceEvent {
            id: "pipeline.start".to_string(),
            message: "build started".to_string(),
            timestamp: SystemTime::now(),
        });

        let graph = model.build(ctx.backend, ctx.trace);

        ctx.trace.record(TraceEvent {
            id: "pipeline.finish".to_string(),
            message: format!("build completed hash={}", graph.stable_hash()),
            timestamp: SystemTime::now(),
        });

        Ok(graph)
    }

    pub fn build_with_validation<M>(
        model: &M,
        ctx: BuildContext<'_>,
    ) -> AphelionResult<BuildGraph>
    where
        M: ModelBuilder<Output = BuildGraph> + ConfigSpec,
    {
        let config = model.config();
        if config.name.trim().is_empty() {
            return Err(AphelionError::InvalidConfig(
                "name cannot be empty".to_string(),
            ));
        }
        if config.version.trim().is_empty() {
            return Err(AphelionError::InvalidConfig(
                "version cannot be empty".to_string(),
            ));
        }

        ctx.trace.record(TraceEvent {
            id: "pipeline.validate".to_string(),
            message: format!("validated {}@{}", config.name, config.version),
            timestamp: SystemTime::now(),
        });

        Self::build(model, ctx)
    }
}
