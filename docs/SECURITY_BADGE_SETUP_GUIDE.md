# Repository Security & Badge Setup Guide

A reusable prompt and schema for implementing security automation and badge visibility on Rust/Python projects.

---

## Prompt for AI Assistant

```
Set up comprehensive security automation and badge visibility for this repository:

1. **README Badges** - Add status badges at the top of README.md:
   - CI workflow status (GitHub Actions)
   - Security audit workflow status
   - crates.io version and downloads (if Rust)
   - docs.rs documentation (if Rust)
   - PyPI version (if Python bindings)
   - License badge
   - MSRV badge (if Rust)

2. **Dependabot** - Create .github/dependabot.yml for automated dependency updates:
   - Cargo ecosystem (weekly)
   - GitHub Actions (weekly)
   - pip/Python (if applicable, weekly)
   - Group minor/patch updates

3. **Security Policy** - Create SECURITY.md with:
   - Supported versions table
   - Vulnerability reporting process
   - Expected response timeline
   - Scope definition

4. **Security Workflow** - Create .github/workflows/security.yml:
   - cargo-audit with SARIF upload to GitHub Security tab
   - cargo-deny for license/advisory/ban checks
   - Scheduled weekly runs + on dependency changes
   - Required permissions: contents: read, security-events: write

5. **Cargo Deny Config** - Create deny.toml:
   - Advisory database configuration
   - License allowlist (MIT, Apache-2.0, BSD variants, etc.)
   - Multiple version warnings
   - Source restrictions

6. **Enable GitHub Security Features** via gh CLI:
   - Private vulnerability reporting
   - Dependabot security updates
   - Secret scanning

Verify all configurations work by running cargo audit and cargo deny locally.
```

---

## File Schema

### 1. README.md Badge Section

```markdown
# Project Name

[![CI](https://github.com/{owner}/{repo}/actions/workflows/ci.yml/badge.svg)](https://github.com/{owner}/{repo}/actions/workflows/ci.yml)
[![Security Audit](https://github.com/{owner}/{repo}/actions/workflows/security.yml/badge.svg)](https://github.com/{owner}/{repo}/actions/workflows/security.yml)
[![crates.io](https://img.shields.io/crates/v/{crate-name}.svg)](https://crates.io/crates/{crate-name})
[![docs.rs](https://docs.rs/{crate-name}/badge.svg)](https://docs.rs/{crate-name})
[![PyPI](https://img.shields.io/pypi/v/{pypi-name}.svg)](https://pypi.org/project/{pypi-name}/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![MSRV](https://img.shields.io/badge/MSRV-{version}-blue.svg)](https://blog.rust-lang.org/)
```

### 2. .github/dependabot.yml

```yaml
version: 2
updates:
  - package-ecosystem: "cargo"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 10
    groups:
      rust-minor:
        patterns: ["*"]
        update-types: ["minor", "patch"]
    commit-message:
      prefix: "deps(rust):"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 5
    commit-message:
      prefix: "deps(actions):"

  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 5
    commit-message:
      prefix: "deps(python):"
```

### 3. SECURITY.md

```markdown
# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

1. **Do NOT** create a public GitHub issue
2. Use GitHub's private vulnerability reporting (Security tab → Report a vulnerability)
3. Or email maintainers directly (see Cargo.toml)

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: 48 hours
- **Assessment**: 7 days
- **Critical fixes**: Prioritized immediately

## Security Measures

- `cargo-audit`: RustSec Advisory Database scanning
- `cargo-deny`: License and dependency auditing
- Dependabot: Automated dependency updates
- CI/CD: All PRs scanned before merge

## Scope

**In scope:**
- Memory safety issues
- Unsafe code vulnerabilities
- Dependency vulnerabilities
- Build process security

**Out of scope:**
- DoS via legitimate API usage
- Third-party dependency issues (report upstream)
```

### 4. .github/workflows/security.yml

```yaml
name: Security Audit

on:
  push:
    branches: [main, master]
    paths:
      - '**/Cargo.toml'
      - '**/Cargo.lock'
  pull_request:
    branches: [main, master]
    paths:
      - '**/Cargo.toml'
      - '**/Cargo.lock'
  schedule:
    - cron: '0 0 * * 1'  # Weekly Monday 00:00 UTC
  workflow_dispatch:

permissions:
  contents: read
  security-events: write

jobs:
  audit:
    name: Security Audit
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-action@stable

      - name: Install cargo-audit
        run: cargo install cargo-audit --locked

      - name: Generate SARIF report
        run: |
          cargo audit --json > audit-results.json || true
          cargo audit --json 2>/dev/null | jq '{
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
              "tool": {
                "driver": {
                  "name": "cargo-audit",
                  "informationUri": "https://github.com/rustsec/rustsec",
                  "rules": []
                }
              },
              "results": [.vulnerabilities.list[]? | {
                "ruleId": .advisory.id,
                "message": { "text": .advisory.title },
                "level": (if .advisory.severity == "critical" then "error" elif .advisory.severity == "high" then "error" else "warning" end),
                "locations": [{
                  "physicalLocation": {
                    "artifactLocation": { "uri": "Cargo.lock" }
                  }
                }]
              }]
            }]
          }' > cargo-audit-results.sarif || echo '{"version":"2.1.0","$schema":"https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json","runs":[{"tool":{"driver":{"name":"cargo-audit"}},"results":[]}]}' > cargo-audit-results.sarif

      - name: Upload SARIF to GitHub Security tab
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: cargo-audit-results.sarif
          category: cargo-audit
        if: always()

      - name: Run cargo-audit (fail on vulnerabilities)
        run: cargo audit --deny warnings

  deny:
    name: Cargo Deny
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: taiki-e/install-action@cargo-deny

      - name: Check advisories
        run: cargo deny check advisories

      - name: Check licenses
        run: cargo deny check licenses

      - name: Check bans
        run: cargo deny check bans
```

### 5. deny.toml

```toml
[advisories]
db-path = "~/.cargo/advisory-db"
db-urls = ["https://github.com/rustsec/advisory-db"]
ignore = [
    # Add RUSSTECs here with comments explaining why
]

[licenses]
confidence-threshold = 0.8
allow = [
    "MIT",
    "Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unicode-DFS-2016",
    "Unicode-3.0",
    "Zlib",
    "CC0-1.0",
    "MPL-2.0",
    "BSL-1.0",
]

[[licenses.clarify]]
name = "ring"
expression = "MIT AND ISC AND OpenSSL"
license-files = [{ path = "LICENSE", hash = 0xbd0eed23 }]

[bans]
multiple-versions = "warn"
wildcards = "warn"
highlight = "all"
workspace-default-features = "allow"
external-default-features = "allow"
skip = []
skip-tree = []

[sources]
unknown-registry = "warn"
unknown-git = "warn"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
```

---

## GitHub CLI Commands

Enable security features programmatically:

```bash
# Enable private vulnerability reporting
gh api repos/{owner}/{repo}/private-vulnerability-reporting -X PUT

# Check current security settings
gh api repos/{owner}/{repo} --jq '.security_and_analysis'

# Enable Dependabot alerts (usually enabled by default for public repos)
gh api repos/{owner}/{repo}/vulnerability-alerts -X PUT
```

---

## Verification Commands

```bash
# Test cargo-audit
cargo audit

# Test cargo-deny (all checks)
cargo deny check

# Test individual deny checks
cargo deny check advisories
cargo deny check licenses
cargo deny check bans
cargo deny check sources
```

---

## Patch Dependencies for Unmaintained Crates

When an upstream dependency uses an unmaintained crate, use `[patch]`:

```toml
# In Cargo.toml
[patch.crates-io]
# Replace unmaintained crate with maintained fork
# Fork must have same crate name in its Cargo.toml
unmaintained-crate = { git = "https://github.com/{owner}/maintained-fork", branch = "patch-target" }
```

**Important**: The fork's `Cargo.toml` must have `name = "unmaintained-crate"` (matching the original) for the patch to work. Create a dedicated branch for this if the fork is published under a different name.

---

## Badge Variants

### shields.io Custom Badges

```markdown
![Custom](https://img.shields.io/badge/{label}-{message}-{color})
![Version](https://img.shields.io/badge/version-1.0.0-green)
![Status](https://img.shields.io/badge/status-stable-brightgreen)
```

### GitHub Actions Dynamic Badges

```markdown
[![Tests](https://github.com/{owner}/{repo}/actions/workflows/{workflow}.yml/badge.svg?branch={branch})](...)
```

### Crates.io Variants

```markdown
[![Downloads](https://img.shields.io/crates/d/{crate}.svg)](https://crates.io/crates/{crate})
[![License](https://img.shields.io/crates/l/{crate}.svg)](https://crates.io/crates/{crate})
```
