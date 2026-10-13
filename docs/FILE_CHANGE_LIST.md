# Proposed file changes versus remote main

Base: `23b1b8a24120420b507815d7cd7035c3f13e17fa` (`origin/main` when cloned). Local branch: `patterns-real-code-prep-20261012`.

- Replace root README with a truthful real-source inventory and reproducibility limitations.
- Add `.gitignore`, `AUDIT_REPORT.md`, `REPRODUCIBILITY.md`, `docs/SOURCE_MAP.md`, and this pre-publication review material.
- Add 38 selected original Python source files under `research_code/`, four historical YAMLs with path-only edits, a selected historical dependency list, an Apache license copy and upstream notice.
- Keep every existing remote file; make no deletion or history rewrite. The nested synthetic demo remains labeled as such.
- No model weights, original datasets, full logs, unrelated source trees, or large binary artifacts are proposed for addition.

The exact staged-file list and diff summary are provided by `git diff --cached --name-status` and `git diff --cached --stat` in this branch. Original-to-release file mapping and SHA-256 hashes are in `SOURCE_MAP.md`.
