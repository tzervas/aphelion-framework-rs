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

## Security Best Practices for Users

When using Aphelion Framework:

1. Keep dependencies up to date with `cargo update`
2. Run `cargo audit` regularly on your projects
3. Use the latest stable Rust compiler
4. Review any unsafe code blocks when contributing
