# aphelion-examples

Example code and demos for the [Aphelion Framework](https://github.com/tzervas/aphelion-framework-rs).

> **Note**: This crate is not published to crates.io. It's for development reference only.

## Running Examples

```bash
# Basic framework demo
cargo run --package aphelion-examples --example demo

# Ternary acceleration demo (BitNet b1.58 + VSA compression)
cargo run --package aphelion-examples --example tritter_demo --features tritter-accel
```

## Example Modules

| Module | Description |
|--------|-------------|
| `basic_usage` | Simple model building with `BuildGraph` and `BuildPipeline` |
| `custom_backend` | Implementing custom `Backend` trait with device capabilities |
| `pipeline_stages` | Creating custom `PipelineStage` implementations |
| `validation` | Using validators to ensure configuration correctness |

## Binary Examples

| Example | Features | Description |
|---------|----------|-------------|
| `demo` | - | Basic framework demo with visual output |
| `tritter_demo` | `tritter-accel` | BitNet b1.58 ternary inference and VSA gradient compression |

## Development

Run tests:

```bash
cargo test --package aphelion-examples
```

Run with all features:

```bash
cargo test --package aphelion-examples --features tritter-accel
```

## See Also

- [Framework README](https://github.com/tzervas/aphelion-framework-rs) - Full documentation
- [API Guide](https://github.com/tzervas/aphelion-framework-rs/blob/main/docs/API-GUIDE.md) - Usage patterns
