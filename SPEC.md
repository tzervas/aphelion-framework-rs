# Aphelion Framework Spec (Root)

This is the canonical, high-level specification and success criteria for the project.
For the expanded version with development notes, see [docs/spec.md](docs/spec.md).

## 1) Purpose
A Rust AI framework that provides:
- Transparent, traceable model construction
- High performance via backend abstraction (burn first)
- Easy configuration with deep customization
- Reproducible builds and deterministic graph hashing
- Integration-ready with rust-ai-core and cubecl

## 2) Core Modules (Canonical)
- `config`: versioned model configs and parameter maps
- `graph`: traceable build graph + deterministic hash
- `diagnostics`: trace events and sinks
- `backend`: backend abstraction (burn-first, cubecl later)
- `pipeline`: build orchestration and validation
- `error`: typed errors
- `macros`: model ergonomics and auto-impls

## 3) Success Criteria (Hybrid: Tests + Docs)
Each criterion is **test-backed** where possible and **documented**.

### SC-1: Deterministic graph hash
- **Test**: `graph_hash_is_deterministic`
- **Docs**: Graph hash policy in [docs/spec.md](docs/spec.md)

### SC-2: Config validation enforced
- **Test**: `pipeline_validates_config`
- **Docs**: Validation rules in [docs/spec.md](docs/spec.md)

### SC-3: Trace capture works
- **Test**: `trace_sink_records_events`
- **Docs**: Trace event schema in [docs/spec.md](docs/spec.md)

### SC-4: Macro ergonomic contract
- **Test**: `macro_build_graph_convention`
- **Docs**: Macro contract in [docs/spec.md](docs/spec.md)

### SC-5: Burn-first backend availability
- **Test**: `burn_backend_config_defaults`
- **Docs**: Burn backend requirements in [docs/spec.md](docs/spec.md)

## 4) Compliance Checklist (v1.0.0)
- [x] SC-1 pass + documented
- [x] SC-2 pass + documented
- [x] SC-3 pass + documented
- [x] SC-4 pass + documented
- [x] SC-5 pass + documented

## 5) Success Criteria v1.1.0 (Developer Experience)

### SC-6: Trace helper methods
- **Test**: `trace_helper_methods_work`
- **Contract**: `ctx.trace.info(id, msg)` reduces 8-line TraceEvent to 1 line

### SC-7: Typed config parameter getters
- **Test**: `typed_config_params`
- **Contract**: `config.param::<T>(key)` with compile-time type safety

### SC-8: Preset pipelines
- **Test**: `preset_pipelines_available`
- **Contract**: `BuildPipeline::standard()`, `::for_training()`, `::for_inference()`

### SC-9: BuildContext ergonomic constructors
- **Test**: `build_context_constructors`
- **Contract**: `BuildContext::new()` eliminates struct literal ceremony

### SC-10: Rich error context and chaining
- **Test**: `errors_have_context`
- **Contract**: Errors include stage, field, and source chain via `std::error::Error`

### SC-11: Getting started documentation
- **Review**: Manual (README quick-start, ARCHITECTURE.md, API-GUIDE.md)
- **Contract**: Zero-to-running in 5 minutes for competent Rust developer

### SC-12: Backend auto-detection
- **Test**: `backend_auto_detect`
- **Contract**: `Backend::auto_detect()` returns best available backend

### SC-13: Async pipeline execution
- **Test**: `async_pipeline_executes`
- **Contract**: `pipeline.execute_async().await` with tokio runtime

## 6) Quality Standards

> *"Talk is cheap. Show me the code."* — Linus Torvalds
> *"We can only see a short distance ahead, but we can see plenty there that needs to be done."* — Alan Turing
> *"The present is theirs; the future, for which I really worked, is mine."* — Nikola Tesla

### Code Quality (Torvalds)
- No unwrap() in library code. Explicit error handling always.
- Commit messages explain WHY, not just WHAT.
- If it's not tested, it doesn't work.
- Premature abstraction is the root of all evil.

### Correctness (Turing)
- Type signatures are contracts. Document invariants.
- Computational complexity documented for non-trivial algorithms.
- Deterministic behavior by default. Randomness is opt-in.
- Edge cases tested, not assumed.

### Design (Tesla)
- APIs should be impossible to misuse.
- Defaults should be safe and sensible.
- Extensibility through composition, not inheritance.
- Today's design enables tomorrow's features.

## 7) Compliance Checklist (v1.1.0)
- [x] SC-6 pass + documented
- [x] SC-7 pass + documented
- [x] SC-8 pass + documented
- [x] SC-9 pass + documented
- [x] SC-10 pass + documented
- [x] SC-11 pass + documented
- [x] SC-12 pass + documented
- [x] SC-13 pass + documented
