# Gemma Teach
## AI teachers can use, trust, and shape.

> ### Teaching is a human craft.
>
> Gemma Teach reimagines AI as something teachers can **shape**, **trust**, and **make their own**.

## Overview

A Claude Code–style harness for teachers, powered by Gemma 3n E2B, running fully on-device.

Gemma Teach is to the classroom what Claude Code is to the terminal: a slash-command interface that turns a language model into a useful collaborator on a fixed pipeline of bounded, validated tasks. Teachers add students, plan lessons, and produce personalized homework without anything leaving the laptop.

See `docs/whitepaper.md` for the architecture write-up and `samples/showcase/fractions-diego/` for an end-to-end run.

## Why this exists

Frontier LLMs are powerful but require sending student-identifying data to a remote API — a non-starter under most schools' privacy expectations. Small local models exist, but the agentic scaffolds designed for frontier models break on a 2 B-parameter model. Gemma Teach inverts that: it ships a scaffold designed *for* Gemma 3n E2B. Each agent session does one small bounded thing, every model output is parsed by a quirk-aware deterministic parser, every step is followed by a validator that enforces the prompt contract. The result is a system you can trust to produce a printable PDF.

## Features

- `/student-add` — open a 5-field modal (name, age & grade, interests, hobbies & media, learning notes), digest it into a structured `student.md` and a `tags.json` of normalized interests, and compute tag overlap with the rest of the class.
- `/class-plan <chapter>` — OCR a textbook chapter (PDF) or load a `.txt`, draft a master class-notes sheet and a master homework with deterministic concept-to-problem mapping, then for each student in the notebook plan a per-student tailoring of named anchors and rewrite the homework around those anchors. Compile everything to PDF via Typst.
- `/student-edit <name>` — apply targeted edits to a student's profile via the `Edit` tool, then regenerate `tags.json` from the updated profile.

## Architecture (workspace)

```
crates/
  gt-core    engine: inference abstraction, parser, quality monitor, skills, sessions
  gt-tools   Read / Write / Edit + OCR (Tesseract) + PDF (Typst) runners
  gt-flows   the three feature pipelines + orchestrator (decomposed agent steps)
  gt-tui     macOS terminal frontend (ratatui)
  gt-ffi     uniffi-rs bindings for future native frontends
skills/      per-tool skill cards + domain knowledge sheets
templates/   Typst templates for PDFs
docs/        whitepaper, decomposition spec, planning artifacts
samples/     showcase: end-to-end inputs and outputs
```

`gt-core` depends on no sibling crate — that is the iOS portability invariant. The Mac frontend and the future Swift app both consume the engine through the same FFI-safe surface.

## Requirements

- macOS (Apple Silicon) with Xcode CLT
- Rust stable (see `rust-toolchain.toml`)
- `tesseract`, `pdftoppm` (`brew install tesseract poppler`)
- `typst` (`brew install typst`)

## Quick start

```sh
cargo run -p gt-tui                  # launches the TUI; on first run, downloads ~3.5 GB Gemma 3n E2B
```

Notebook lives at `~/GemmaTeach/`. Model cache at `~/.gemma-teach/models/`. Logs at `~/.gemma-teach/logs/gemma-teach-<date>.log` — `tail -f` that file for inference progress while the TUI is open.

## Showcase

`samples/showcase/fractions-diego/` contains a complete end-to-end run against the real Gemma 3n model: a math chapter on fractions and ratios, a 6th-grade Barcelona-FC obsessed student profile for "Diego," the master class-notes PDF and master homework PDF that apply to the whole class, the tailoring plan the system picks for Diego specifically (Barcelona match scores, Dragon Ball Z power levels), and Diego's personalized homework PDF where the math problems are reformulated so the named-interest scenarios supply the numerical operands — *"Barcelona scored 2 goals out of 3 shots in the first half. What is the fraction of goals scored compared to the number of shots taken?"* in place of *"Find an equivalent fraction for 2/3."* Files are numbered in the order the pipeline produced them.

## Testing

```sh
cargo test --workspace                              # unit + integration, 98 tests, no model needed
GEMMA_TEACH_SMOKE=1 cargo test --features smoke -p gt-flows  # real-model smoke
cargo run -p gt-flows --example record_trace --features smoke -- student-add --name Maya --description "..." --out traces/<file>.jsonl
```

The `record_trace` example writes every flow and session event as JSONL so traces can drive prompt and parser iteration. `traces/phase-2-*.jsonl` are the recorded runs that produced the current system.

## Status

Phase 1 (macOS TUI, all three flows end-to-end) ships in this repository. Phase 2 (native frontends via `uniffi-rs`) is scaffolded behind `gt-ffi` and ready to wire up. `docs/whitepaper.md` documents the architecture and the design choices in depth; `docs/tailor-decomposition.md` specifies the next round of per-concept micro-decomposition that would unlock tailored class-notes.
