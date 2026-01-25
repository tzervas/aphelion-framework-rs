# Aphelion Framework 1.0.0 Roadmap

## Current State (v0.1.0)
- 4/5 success criteria tests passing
- Core APIs scaffolded: config, graph, backend, pipeline, diagnostics
- Proc macro: #[aphelion_model] working
- Feature flags defined for ecosystem integrations

## 1.0.0 Release Requirements

### Phase 1: Core Completion (Non-breaking, parallel-safe)

#### Branch: feature/error-handling
- [ ] Expand AphelionError with more variants (Validation, Serialization, IO)
- [ ] Add error context/source chain support
- [ ] Add From impls for common error types
- [ ] Unit tests for error handling

#### Branch: feature/config-validation
- [ ] Add ConfigValidator trait
- [ ] Implement default validators (name, version, params schema)
- [ ] Add custom validator support
- [ ] Validation error messages with context
- [ ] Tests for validation edge cases

#### Branch: feature/graph-enhancements
- [ ] Add graph traversal methods (topological sort, cycle detection)
- [ ] Add node metadata support
- [ ] Add edge weights/labels
- [ ] Graph serialization (serde)
- [ ] Graph visualization prep (DOT format export)

#### Branch: feature/diagnostics-export
- [ ] JSON trace exporter
- [ ] Trace filtering by level
- [ ] Trace timestamps with configurable format
- [ ] Trace sink composition (multi-sink)
- [ ] OpenTelemetry prep (span IDs, trace context)

### Phase 2: Pipeline & Backend (Sequential after Phase 1)

#### Branch: feature/pipeline-stages
- [ ] Define PipelineStage trait
- [ ] Pre-build hooks
- [ ] Post-build hooks
- [ ] Stage skipping/filtering
- [ ] Pipeline progress reporting

#### Branch: feature/backend-abstraction
- [ ] Refine Backend trait with lifecycle methods
- [ ] Device capability queries
- [ ] Memory management interface
- [ ] Backend registry/factory pattern
- [ ] MockBackend for comprehensive testing

### Phase 3: Macro & Ergonomics (After Phase 1)

#### Branch: feature/macro-derive
- [ ] Add #[derive(AphelionConfig)] for auto ConfigSpec
- [ ] Better compile error messages
- [ ] Attribute options (#[aphelion(skip)], #[aphelion(rename)])
- [ ] Support for generics in models

#### Branch: feature/builder-pattern
- [ ] Fluent builder for ModelConfig
- [ ] Type-safe parameter builders
- [ ] Config presets/templates

### Phase 4: Documentation & Examples (Parallel with all)

#### Branch: feature/api-docs
- [ ] Rustdoc for all public items
- [ ] Module-level documentation
- [ ] Code examples in docs
- [ ] Panic/error documentation

#### Branch: feature/examples
- [ ] Basic usage example
- [ ] Custom backend example
- [ ] Trace export example
- [ ] Multi-stage pipeline example

### Phase 5: Testing & CI

#### Branch: feature/test-coverage
- [ ] Property-based tests for determinism
- [ ] Fuzzing for config parsing
- [ ] Benchmark suite
- [ ] Integration test suite expansion

#### Branch: feature/ci-setup
- [ ] GitHub Actions workflow
- [ ] Clippy + rustfmt checks
- [ ] Test matrix (stable, nightly, MSRV)
- [ ] Coverage reporting

## Branch Dependencies

```
main
  └── develop
        ├── feature/error-handling (Phase 1 - parallel)
        ├── feature/config-validation (Phase 1 - parallel)
        ├── feature/graph-enhancements (Phase 1 - parallel)
        ├── feature/diagnostics-export (Phase 1 - parallel)
        ├── feature/api-docs (Phase 4 - parallel with all)
        │
        ├── feature/pipeline-stages (Phase 2 - after Phase 1)
        ├── feature/backend-abstraction (Phase 2 - after Phase 1)
        │
        ├── feature/macro-derive (Phase 3 - after Phase 1)
        ├── feature/builder-pattern (Phase 3 - after Phase 1)
        │
        ├── feature/examples (Phase 4 - after Phase 2)
        ├── feature/test-coverage (Phase 5 - after Phase 3)
        └── feature/ci-setup (Phase 5 - independent)
```

## Success Criteria for 1.0.0

1. All 5 SC tests passing with documentation
2. Zero clippy warnings
3. 80%+ test coverage on core modules
4. Complete rustdoc for public API
5. At least 3 working examples
6. CI pipeline green
7. CHANGELOG.md with all changes
8. Version bump to 1.0.0

## Agent Assignment Strategy

Phase 1 branches can be worked in parallel by separate agents:
- Agent A: error-handling + config-validation (related concerns)
- Agent B: graph-enhancements (isolated module)
- Agent C: diagnostics-export (isolated module)
- Agent D: api-docs (can work independently)

Phase 2+ must wait for Phase 1 merges to develop.
