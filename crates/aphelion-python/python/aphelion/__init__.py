"""
Aphelion AI Framework - Python Bindings

This package provides Python bindings for the Aphelion AI framework,
a Rust-based framework for building and managing AI models.

Example:
    >>> import aphelion
    >>> import asyncio
    >>>
    >>> async def main():
    ...     config = aphelion.ModelConfig("my-model", "1.0.0")
    ...     graph = aphelion.BuildGraph()
    ...     node = graph.add_node("layer", config)
    ...     
    ...     ctx = aphelion.BuildContext.with_null_backend()
    ...     pipeline = aphelion.BuildPipeline.standard()
    ...     result = await pipeline.execute_async(ctx, graph)
    ...     print(f"Hash: {result.stable_hash()}")
    >>>
    >>> asyncio.run(main())
"""

from .aphelion import (
    # Version
    __version__,
    
    # Feature flags
    HAS_BURN,
    HAS_CUBECL,
    HAS_RUST_AI_CORE,
    HAS_TRITTER_ACCEL,
    
    # Configuration
    ModelConfig,
    ModelConfigBuilder,
    
    # Graph
    NodeId,
    GraphNode,
    BuildGraph,
    
    # Backend
    NullBackend,
    
    # Diagnostics
    TraceLevel,
    TraceEvent,
    InMemoryTraceSink,
    
    # Pipeline
    BuildContext,
    BuildPipeline,
    
    # Validation
    ValidationError,
    NameValidator,
    VersionValidator,
    CompositeValidator,
)

__all__ = [
    # Version
    "__version__",
    
    # Feature flags
    "HAS_BURN",
    "HAS_CUBECL",
    "HAS_RUST_AI_CORE",
    "HAS_TRITTER_ACCEL",
    
    # Configuration
    "ModelConfig",
    "ModelConfigBuilder",
    
    # Graph
    "NodeId",
    "GraphNode",
    "BuildGraph",
    
    # Backend
    "NullBackend",
    
    # Diagnostics
    "TraceLevel",
    "TraceEvent",
    "InMemoryTraceSink",
    
    # Pipeline
    "BuildContext",
    "BuildPipeline",
    
    # Validation
    "ValidationError",
    "NameValidator",
    "VersionValidator",
    "CompositeValidator",
]
