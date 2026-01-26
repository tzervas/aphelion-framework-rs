# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Aphelion Framework, please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Email security concerns to the maintainers directly (see [Cargo.toml](./Cargo.toml) for contact info)
3. Include as much detail as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours of your report
- **Assessment**: Within 7 days, we'll provide an initial assessment
- **Resolution**: Critical vulnerabilities will be prioritized for immediate patching
- **Disclosure**: We follow responsible disclosure practices

### Security Measures

This project employs several automated security measures:

- **`cargo-audit`**: Automated vulnerability scanning against the RustSec Advisory Database
- **Dependabot**: Automated dependency updates for Rust, Python, and GitHub Actions
- **CI/CD Security**: All PRs are scanned before merge

### Scope

The following are in scope for security reports:

- Memory safety issues
- Cryptographic vulnerabilities (if applicable)
- Unsafe code blocks that could lead to undefined behavior
- Dependency vulnerabilities
- Build process security issues

### Out of Scope

- Denial of service via legitimate API usage
- Issues in third-party dependencies (report these upstream)
- Social engineering attacks

## Unmaintained Dependency Mitigation

### Background

Several transitive dependencies in the Rust AI ecosystem are currently unmaintained:

| Crate | Status | Advisory |
|-------|--------|----------|
| `paste` | Unmaintained | [RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436) |
| `gemm` | Unmaintained | [RUSTSEC-2024-0428](https://rustsec.org/advisories/RUSTSEC-2024-0428) |

These crates are transitive dependencies via `candle-core` (used by the Rust AI ecosystem including `burn` and `cubecl`).

### Maintained Forks

We maintain actively patched forks of these dependencies:

| Original | Maintained Fork | Version | Repository |
|----------|-----------------|---------|------------|
| `paste` | `qlora-paste` | 1.0.20+ | [github.com/tzervas/qlora-paste](https://github.com/tzervas/qlora-paste) |
| `gemm` | `qlora-gemm` | 0.20.0+ | [github.com/tzervas/qlora-gemm](https://github.com/tzervas/qlora-gemm) |
| `candle-core` | `qlora-candle` | 0.9.x | [github.com/tzervas/qlora-candle](https://github.com/tzervas/qlora-candle) |

### Upstream Contribution

We have submitted a PR to merge these fixes upstream to HuggingFace Candle:

- **PR**: [huggingface/candle#3335](https://github.com/huggingface/candle/pull/3335)
- **Status**: Pending review
- **Impact**: Once merged, all candle-dependent projects can use maintained dependencies

### For Other Projects

If your project depends on `candle-core` and you want to use maintained dependencies, add this to your `Cargo.toml`:

```toml
[patch.crates-io]
candle-core = { git = "https://github.com/tzervas/qlora-candle.git", branch = "use-qlora-gemm" }
```

This patches the entire dependency tree to use:
- `qlora-gemm` instead of `gemm`
- `qlora-paste` instead of `paste`

### Dependency Chain

```
candle-core (qlora-candle)
└── qlora-gemm v0.20.0
    └── qlora-paste v1.0.20
```

## Security Best Practices for Users

When using Aphelion Framework:

1. Keep dependencies up to date with `cargo update`
2. Run `cargo audit` regularly on your projects
3. Use the latest stable Rust compiler
4. Review any unsafe code blocks when contributing
5. Use the maintained fork patches if security audits flag `paste` or `gemm`
