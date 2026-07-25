# Versioning and releases

## Current version

**`1.2.10`** — and that number is real. Unlike most of this fleet, `aphelion-framework-rs` is
**genuinely published to crates.io**:

| | Version |
| --- | --- |
| `[workspace.package] version` | `1.2.10` |
| Newest git tag | `v1.2.10` |
| **crates.io `aphelion-core`** | **`1.2.10`** |
| **crates.io `aphelion-macros`** | **`1.2.10`** |

Everything agrees, and real dependents can resolve those artifacts. That fact drives every rule
below.

## This repo does **not** use `major_version_zero` — and must not

The fleet convention is `major_version_zero = true` in every `.cz.toml`, because nearly every
repo in the fleet is 0.x and that key is what keeps it there. **This repo is the exception.**

`major_version_zero` does not stop applying once a project passes 1.0. It pins the major
*permanently*, and it does so by demoting breaking changes to MINOR bumps. Measured on a fixture
at version 1.2.10:

```
BREAKING @1.2.10, major_version_zero = true   ->  bump: 1.2.10 -> 1.3.0   (MINOR)
BREAKING @1.2.10, major_version_zero absent   ->  bump: 1.2.10 -> 2.0.0   (MAJOR)
```

A cargo dependent on `aphelion-core = "1.2"` accepts anything `>=1.2.0, <2.0.0`. Shipping a
breaking change as `1.3.0` would therefore land it **silently in every consumer's next build** —
a semver violation against published artifacts that cannot be unpublished, only yanked.

So: do not add that key here. The `.cz.toml` carries the same warning inline.

## At 1.x, MAJOR is the breaking position

| Change                        | Bump      | Example            |
| ----------------------------- | --------- | ------------------ |
| `fix:`                        | PATCH     | 1.2.10 → 1.2.11    |
| `feat:`                       | MINOR     | 1.2.10 → 1.3.0     |
| `feat!:` / `BREAKING CHANGE:` | **MAJOR** | 1.2.10 → **2.0.0** |

This is ordinary semver, and it is the *opposite* of the 0.x repos in this fleet, where the major
is pinned and MINOR carries breaking changes. If you move between repos here, check which regime
you are in before reasoning about a bump.

Consumers pin the **major**: `aphelion-core = "1.2"` (equivalently `^1.2`) tracks every compatible
release up to but excluding `2.0.0`.

### A major bump still needs a human

`cz bump` will happily compute `2.0.0` from a `feat!:` commit. **No agent may cut or propose a
2.0.0 release**, exactly as no agent may cut a 1.0.0 elsewhere in the fleet. A major bump against
a published crate breaks every dependent by design; that is a maintainer decision, made
deliberately, with a migration note.

## Version files

The workspace version appears in three places in `Cargo.toml`:

1. `[workspace.package] version`
2. `[workspace.dependencies] aphelion-macros = { version = "…", path = … }`
3. `[workspace.dependencies] aphelion-core   = { version = "…", path = … }`

All three are covered by the single `version_files` entry `"Cargo.toml:version = "`. Verified with
a real bump — `1.2.10 → 2.0.0` moved exactly those three lines and left third-party pins
(`serde = { version = "1.0" }`, `tokio = { version = "1" }`) untouched, because cz only replaces
the literal current version string.

The per-crate manifests under `crates/*/Cargo.toml` use `version = { workspace = true }` and
inherit, so they hold no copy of their own.

Do not hand-edit a version — run the tool:

```bash
cz bump --yes --dry-run     # show what would happen, change nothing
cz bump                     # move every version file + create the tag
cz version --project        # what this project currently claims to be
```

Refresh `Cargo.lock` with `cargo build` after a bump.

## Documentation version references

Two different kinds of number appear in the docs; only one of them tracks releases.

- **Dependency examples** — `aphelion-core = { version = "1.2", … }` — *do* track the current
  minor and should be kept consistent. They are prose, not parsed, so they are not in
  `version_files`; check them when you cut a minor.
- **Sample program output** — e.g. the `Version: 1.0.0` line in the README's Model Configuration
  block, and `config.version = "1.0.0"` in `docs/architecture.md` — is a **model/config schema
  version emitted by the example**, not this package's version. Leave it alone. Rewriting it to
  match the crate version would be wrong.

## A GitHub Release is not a registry publication

The two are different things, and this is the one repo in the fleet where both have actually
happened. Keep them distinct anyway:

- **A git tag / GitHub Release** is a marker plus notes. It publishes nothing consumable.
- **A crates.io publication** is the artifact dependents actually resolve.

A tag existing is not evidence that `cargo add aphelion-core@X` works. Check crates.io. When you
claim a version is released, say *where*.

## Release steps

1. Land work on a work branch and open a PR; merge with a **merge commit**, never a squash.
2. `cz bump` on the release branch: this moves every version file and creates the tag locally.
3. Push the tag; the `release` workflow (`workflow_dispatch`) builds the GitHub Release.
4. **Publishing to crates.io is a separate, deliberate step** — `aphelion-macros` first, then
   `aphelion-core`, since the latter depends on the former.
