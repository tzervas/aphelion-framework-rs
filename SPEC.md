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

## 4) Compliance Checklist
- [x] SC-1 pass + documented
- [x] SC-2 pass + documented
- [x] SC-3 pass + documented
- [x] SC-4 pass + documented
- [x] SC-5 pass + documented
