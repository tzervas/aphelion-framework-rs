# Aphelion Framework - Claude Code Instructions

## Project Overview
Aphelion is a Rust AI framework providing transparent, traceable model construction with backend abstraction (burn-first approach). Target: 1.0.0 release.

## Repository Structure
```
aphelion-framework-rs/
├── Cargo.toml              # Workspace root
├── SPEC.md                 # Canonical spec (source of truth)
├── docs/spec.md            # Expanded spec with dev notes
└── crates/
    ├── aphelion-core/      # Core APIs (config, graph, backend, pipeline, diagnostics)
    ├── aphelion-macros/    # #[aphelion_model] proc macro
    ├── aphelion-tests/     # Test suite (SC-1 through SC-5)
    └── aphelion-examples/  # Usage examples
```

## Build Commands
```bash
# Build all crates
cargo build --workspace

# Build with all features (burn, cubecl, rust-ai-core)
cargo build --workspace --all-features

# Run tests
cargo test --workspace

# Run tests with burn feature
cargo test --workspace --features burn

# Check formatting
cargo fmt --check --all

# Lint
cargo clippy --workspace --all-features
```

## Key Design Patterns
- **Trait-based abstraction**: Backend, TraceSink, ModelBuilder, ConfigSpec
- **Feature flags**: `burn`, `cubecl`, `rust-ai-core` for ecosystem integrations
- **Deterministic by design**: BTreeMap for params, SHA256 for graph hashing
- **Macro convention**: `#[aphelion_model]` requires `config` field and `build_graph()` method

## Success Criteria (1.0.0 Requirements)
All must pass tests AND have documentation:
- SC-1: Deterministic graph hash
- SC-2: Config validation enforced
- SC-3: Trace capture works
- SC-4: Macro ergonomic contract
- SC-5: Burn-first backend availability

## Branching Strategy
- `main`: Stable releases only
- `develop`: Integration branch for features
- `feature/*`: Individual feature branches (merge to develop)
- `fix/*`: Bug fixes (merge to develop)

## Code Style
- Follow Rust 2021 idioms
- Use `thiserror` for error types
- Use `tracing` for instrumentation
- Prefer explicit over implicit
- No unwrap() in library code - use proper error handling

## Testing Requirements
- All public APIs must have tests
- Tests should be in `aphelion-tests` crate for integration tests
- Unit tests can be inline in modules
- Feature-gated code needs feature-gated tests

## Resource Constraints
- **CRITICAL**: Minimize /tmp usage - clean up temp files promptly
- Use workspace target directory, not /tmp for builds
- Avoid large intermediate artifacts

## Agent Instructions
When implementing features:
1. Read SPEC.md and docs/spec.md first
2. Check existing patterns in aphelion-core
3. Follow trait-based abstraction pattern
4. Add tests to aphelion-tests
5. Update docs/spec.md with implementation notes
6. Keep PRs focused - one feature per branch
