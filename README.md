# Gemma Teach

A Claude Code–style harness for teachers, powered by Gemma 3n E2B, fully offline.

Gemma Teach helps teachers make material more accessible and personally tailored to each student, understand a class's weaknesses, and plan lessons — all on-device. No student information leaves the teacher's machine.

This repository hosts the Rust engine and the macOS terminal frontend. An iPhone app is planned (Phase 2) and consumes the same engine via `uniffi-rs`.

## Why this exists

Small local models fail under scaffolds designed for frontier models. Gemma Teach is built around the constraints of a small model: a deterministic output parser, a quality monitor, prescriptive recovery messages, per-turn skill injection, thinking and turn budgets — patterns drawn from [`little-coder`](https://github.com/itayinbarr/little-coder) and the paper [*Honey, I Shrunk the Coding Agent*](https://itayinbarr.substack.com/p/honey-i-shrunk-the-coding-agent).

## Features

- `/student-add` — add a student to the class notebook with their name, date, and interests; auto-extract normalized tags and compute overlaps with the rest of the class.
- `/class-plan <pdf>` — OCR a textbook chapter, generate teacher class notes and a homework sheet, then re-skin both for every student through the lens of their interests; export as PDFs.
- `/student-edit <name>` — update a student's profile and refresh their tags.

## Architecture (workspace)

```
crates/
  gt-core    engine: inference abstraction, parser, quality monitor, skills, sessions
  gt-tools   Read / Write / Edit + OCR (Tesseract) + PDF (Typst) runners
  gt-flows   the three feature pipelines + orchestrator
  gt-tui     macOS terminal frontend (ratatui)
  gt-ffi     uniffi-rs bindings for the future iPhone app
skills/      per-tool skill cards + domain knowledge sheets
templates/   Typst templates for PDFs
```

## Requirements

- macOS (Apple Silicon) with Xcode CLT
- Rust stable (see `rust-toolchain.toml`)
- `tesseract`, `pdftoppm` (`brew install tesseract poppler`)
- `typst` (`brew install typst`)

## Quick start

```sh
cargo run -p gt-tui                  # launches the TUI; on first run, downloads ~3.5 GB Gemma 3n E2B
```

Notebook lives at `~/GemmaTeach/`. Model cache at `~/.gemma-teach/models/`.

## Status

Pre-MVP. See `plans/` and the [implementation plan](../../.claude/plans/i-want-to-build-melodic-hamster.md) for the commit-by-commit sequence.
