# aphelion-tests

Integration tests for the [Aphelion Framework](https://github.com/tzervas/aphelion-framework-rs).

> **Note**: This crate is not published to crates.io. It's for development and CI testing only.

## Running Tests

```bash
# Run all integration tests
cargo test --package aphelion-tests

# With async support
cargo test --package aphelion-tests --features tokio

# With tritter-accel acceleration tests
cargo test --package aphelion-tests --features tritter-accel

# All features
cargo test --package aphelion-tests --all-features
```

## Test Categories

### Core Integration Tests
- Graph building and hashing
- Pipeline execution and hooks
- Configuration validation
- Backend trait implementations

### Feature-Gated Tests

| Feature | Tests |
|---------|-------|
| `tokio` | Async pipeline execution |
| `burn` | Burn backend integration |
| `cubecl` | CubeCL backend integration |
| `tritter-accel` | Ternary acceleration pipeline stages |

## Test Organization

Tests are organized in `src/lib.rs` with sections:

1. **CONFIG TESTS** - ModelConfig, typed params, presets
2. **GRAPH TESTS** - BuildGraph, edges, hashing
3. **PIPELINE TESTS** - Stages, hooks, execution
4. **BACKEND TESTS** - Backend trait, capabilities
5. **DIAGNOSTICS TESTS** - Tracing, events
6. **VALIDATION TESTS** - Validators, composition
7. **FEATURE-GATED TESTS** - Optional backend tests

## CI Integration

These tests run in the CI pipeline:

```yaml
# .github/workflows/ci.yml
- name: Run tests
  run: cargo test --workspace
```

## See Also

- [SPEC.md](https://github.com/tzervas/aphelion-framework-rs/blob/main/SPEC.md) - Success criteria
- [CI Workflow](https://github.com/tzervas/aphelion-framework-rs/blob/main/.github/workflows/ci.yml)
