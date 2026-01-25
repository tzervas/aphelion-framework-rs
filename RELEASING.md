# Release Process

This document describes how to release new versions of Aphelion.

## Prerequisites

### GitHub Secrets

Configure the following secrets in your GitHub repository settings:

1. **CARGO_REGISTRY_TOKEN**
   - Go to [crates.io/settings/tokens](https://crates.io/settings/tokens)
   - Create a new token with "publish-new" and "publish-update" scopes
   - Add as repository secret: Settings → Secrets and variables → Actions → New repository secret

2. **PyPI Trusted Publishing** (recommended)
   - Go to [pypi.org/manage/project/aphelion-framework/settings/publishing/](https://pypi.org/manage/project/aphelion-framework/settings/publishing/)
   - Add a new trusted publisher:
     - Owner: `tzervas`
     - Repository: `aphelion-framework-rs`
     - Workflow: `release.yml`
     - Environment: `pypi`

   Or use **PYPI_API_TOKEN**:
   - Go to [pypi.org/manage/account/token/](https://pypi.org/manage/account/token/)
   - Create a token scoped to the `aphelion-framework` project
   - Add as repository secret

## Version Bump

1. Update versions in:
   - `Cargo.toml` (workspace.package.version)
   - `crates/aphelion-python/pyproject.toml` (project.version)
   - `CHANGELOG.md`

2. Commit the version bump:
   ```bash
   git add -A
   git commit -m "chore: bump version to X.Y.Z"
   ```

## Creating a Release

1. Create and push a version tag:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

2. The release workflow will automatically:
   - Run tests
   - Create a GitHub Release with auto-generated release notes
   - Publish to crates.io (aphelion-macros, aphelion-core, aphelion-examples)
   - Build Python wheels for Linux, macOS, and Windows
   - Publish to PyPI

## Manual Publishing

If you need to publish manually:

### crates.io

```bash
# Publish in order (macros → core → examples)
cargo publish -p aphelion-macros
sleep 30  # Wait for crates.io index
cargo publish -p aphelion-core
sleep 30
cargo publish -p aphelion-examples
```

### PyPI

```bash
cd crates/aphelion-python
pip install maturin twine

# Build wheels
maturin build --release

# Upload to PyPI
twine upload target/wheels/*
```

## Pre-release Versions

For alpha/beta/rc releases, use tags like:
- `v1.0.0-alpha.1`
- `v1.0.0-beta.1`
- `v1.0.0-rc.1`

These will be marked as pre-releases on GitHub.

## Troubleshooting

### crates.io publish fails

- Check that `CARGO_REGISTRY_TOKEN` is set correctly
- Ensure version numbers are correct and not already published
- Verify package metadata in Cargo.toml files

### PyPI publish fails

- Check trusted publishing configuration
- Verify the package name isn't already taken
- Ensure version isn't already published

### Wheel builds fail

- Check Rust toolchain installation on CI
- Verify maturin configuration in pyproject.toml
- Check for platform-specific compilation issues
