# Embedding Gemma Teach in your application

This guide is for developers who want to drive Gemma Teach's flows from their own
software — a school dashboard, an LMS plugin, a desktop app, a batch job — instead
of through the `gt-tui` terminal UI.

## What "SDK" means here today

Be clear-eyed about the current state:

- **There is no packaged, versioned, cross-language SDK yet.** The integration
  surface is the **Rust crates of this workspace** (`gt-core`, `gt-flows`,
  `gt-tools`). If your application is written in Rust, you depend on those crates
  directly — that is the supported path today.
- **`gt-core` is deliberately I/O-free and FFI-clean** (no terminal code, no
  hard-coded paths, all public types are `Send + Sync`). That is what *makes* a
  future cross-language SDK possible.
- **`gt-ffi` is a scaffold, not a shipped SDK.** It exposes one demo function and
  compiles as a C-ABI library; full `uniffi`-generated Swift/Kotlin bindings are
  Phase 2 work and are **not done**. See [Non-Rust applications](#non-rust-applications)
  for what is and isn't possible right now.

If you need to embed Gemma Teach into a non-Rust app *today*, read the caveats in
the last section before committing to it.

## Architecture you integrate against

```
your application
      │
      ├─ gt-flows   ── build a Flow + FlowCtx, run it with the Orchestrator
      ├─ gt-tools   ── Read / Write / Edit tools, Tesseract OCR, Typst PDF
      └─ gt-core    ── LlmBackend (Gemma runtime), parser, quality monitor, sessions
```

You generally touch three things:

1. A **backend** (`Arc<dyn LlmBackend>`) — the model runtime.
2. A **`ToolRegistry`** — the tools agent steps may call.
3. The **`Orchestrator`** — runs a `Flow` and streams events back to you.

Flows write their results to a **notebook directory** on disk (the `root` path you
pass in). Your app reads the produced artifacts back from that directory, or from
the `FlowCtx.artifacts` map returned when the flow finishes.

## 1. Add the dependency

Point at the crates by path (vendored) or git:

```toml
# Cargo.toml
[dependencies]
gt-core  = { git = "https://github.com/itayinbarr/gemma-teach", features = ["backend-llama", "model-fetch"] }
gt-flows = { git = "https://github.com/itayinbarr/gemma-teach" }
gt-tools = { git = "https://github.com/itayinbarr/gemma-teach" }
tokio    = { version = "1", features = ["rt-multi-thread", "macros", "sync"] }
chrono   = "0.4"
```

`gt-core` feature flags:

| Feature | Pulls in | When you need it |
|---|---|---|
| `backend-llama` | `llama-cpp-2` | Running the real Gemma model. Omit for mock-only/test builds. |
| `model-fetch`   | `reqwest`, `sha2` | First-launch model download + SHA-256 verification. |

Without `backend-llama` you can still build and run flows against `MockBackend` /
`EchoBackend` — useful for tests and CI where you don't want a 3.1 GB model.

## 2. Build a backend

**Production — local Gemma:**

```rust
use std::sync::Arc;
use gt_core::backend::LlmBackend;
use gt_core::llama_backend::{LlamaCppBackend, LlamaConfig};

let model_path = "/path/to/gemma-4-E2B-it-Q4_K_M.gguf".into();
let backend: Arc<dyn LlmBackend> =
    Arc::new(LlamaCppBackend::new(LlamaConfig::new(model_path)));
```

`LlamaConfig::new` defaults to `n_ctx = 32_768` and offloads all layers to the GPU
(`n_gpu_layers = 999`, Metal on Apple Silicon). Override the fields if you need a
smaller context or CPU-only inference.

To download the model on first launch (with the `model-fetch` feature):

```rust
use gt_core::model_fetch::{default_models_dir, fetch, FetchSpec};

let dir  = default_models_dir();          // ~/.gemma-teach/models
let spec = FetchSpec::gemma_4_e2b_q4km(&dir);
fetch(spec, None).await?;                 // pass Some(mpsc::Sender) for progress events
```

**Tests / CI — no model:**

```rust
use gt_core::backend::{MockBackend, EchoBackend};
let backend = Arc::new(MockBackend::new());   // or EchoBackend, or MockBackend with a MockScript
```

## 3. Build a tool registry

Agent steps inside flows call `Read` / `Write` / `Edit`. Register them once:

```rust
use std::sync::Arc;
use gt_core::tool::ToolRegistry;

let tools = ToolRegistry::new()
    .register(Arc::new(gt_tools::ReadTool))
    .register(Arc::new(gt_tools::WriteTool))
    .register(Arc::new(gt_tools::EditTool));
```

## 4. Run a flow

All three flows follow the same shape: build `(Flow, FlowCtx)`, hand them to an
`Orchestrator`, then consume the event channels.

### `/student-add`

```rust
use chrono::Local;
use gt_flows::orchestrator::Orchestrator;
use gt_flows::student_add;

let root = std::path::PathBuf::from("/Users/teacher/GemmaTeach");
let date = Local::now().date_naive();

let (flow, ctx) = student_add::flow_with_ctx(
    root,
    date,
    "Diego".to_string(),
    "Obsessed with Barcelona FC, watches Dragon Ball Z, quick with concrete numbers…".to_string(),
);

let handle = Orchestrator::new(backend.clone(), tools.clone()).start(flow, ctx);

// Drive the flow to completion and inspect produced artifacts.
let final_ctx = handle.join.await??;
for (key, path) in final_ctx.artifacts.iter() {
    println!("{key} -> {}", path.display());
}
```

### `/student-edit`

```rust
let (flow, ctx) = gt_flows::student_edit::flow_with_ctx(
    root, date,
    "Diego".to_string(),
    "He moved up a reading group; add a note about chapter-book stamina.".to_string(),
);
let handle = Orchestrator::new(backend.clone(), tools.clone()).start(flow, ctx);
```

### `/class-plan`

`/class-plan` also needs an OCR runner, a PDF runner, and a Typst templates
directory. It tailors homework for **every student already in the notebook**.

```rust
use std::sync::Arc;
use gt_flows::class_plan::{flow_with_ctx_from_source, ClassPlanSource};
use gt_tools::{TesseractRunner, TypstRunner};

let ocr = Arc::new(TesseractRunner::new());   // MockOcrRunner for tests
let pdf = Arc::new(TypstRunner::new());       // MockPdfRunner for tests
let templates_dir = std::path::PathBuf::from("templates/typst");

let source = ClassPlanSource::from_path("samples/chapters/fractions-and-ratios.txt".into());
// or ClassPlanSource::Pdf(path) / ClassPlanSource::Text(String)

let (flow, ctx) = flow_with_ctx_from_source(
    root, date, source, ocr, pdf, templates_dir,
)?;

// Per-student tailoring sessions run in a parallel group; cap concurrency here.
let handle = Orchestrator::new(backend.clone(), tools.clone())
    .with_parallelism(1)   // Gemma 4 on Metal uses the whole GPU per inference
    .start(flow, ctx);
```

## 5. Consume events for progress / streaming

`Orchestrator::start` returns immediately with an `OrchestratorHandle`:

```rust
pub struct OrchestratorHandle {
    pub flow_events:    mpsc::Receiver<FlowEvent>,
    pub session_events: HashMap<StepId, mpsc::Receiver<SessionEvent>>,
    pub join:           JoinHandle<Result<FlowCtx, FlowError>>,
}
```

- **`flow_events`** — pipeline-level: `FlowStarted { steps }`, `StepStateChanged
  { step, state }`, `StepArtifactProduced { step, key, path }`, `FlowDone { ok }`.
  Drive a progress bar or step list from these.
- **`session_events`** — per agent step: `TokenDelta` (stream model output to a
  UI), `ToolCallStarted`, `ToolCallResult`, `Done`, `Failed`.
- **`join`** — awaits the final `FlowCtx`; `Err(FlowError)` means a step or
  validator rejected the run.

```rust
let mut flow_rx = handle.flow_events;
tokio::spawn(async move {
    while let Some(ev) = flow_rx.recv().await {
        match ev {
            FlowEvent::StepStateChanged { step, state } => update_ui(step, state),
            FlowEvent::StepArtifactProduced { key, path, .. } => record(key, path),
            FlowEvent::FlowDone { ok, .. } => mark_done(ok),
            _ => {}
        }
    }
});
let final_ctx = handle.join.await??;
```

A flow that returns `Err` did **not** silently ship bad output — the deterministic
validators (concept-mapping, tailoring divergence, tag shape) fail loudly. Surface
the `FlowError` to your user; do not treat a failed flow as a partial success.

## Where results land

Flows write to the notebook `root` you passed in:

```
<root>/students/<slug>/{student.md, tags.json}
<root>/lessons/<YYYY-MM-DD>/{source.txt, class-notes.md, homework.md,
                             class-notes.pdf, homework.pdf,
                             per-student/<slug>/{tailoring-plan.md, homework.md, homework.pdf}}
```

Your application reads these files back. The `FlowCtx.artifacts` map returned from
`handle.join.await` gives you the key→path index for the artifacts a run produced.

## Non-Rust applications

If your app is **not** Rust, there is no clean integration path today. Honest options:

1. **Wait for Phase 2.** `gt-ffi` is intended to expose the engine via
   `uniffi-rs`-generated Swift (and, in principle, Kotlin) bindings. That work is
   not finished — `gt-ffi` currently exposes a single demo function and runs only
   against mock backends.
2. **Shell out to the CLI.** Treat `gt-tui` (or a thin custom binary built on
   `gt-flows`) as a subprocess, pass inputs, and read the notebook directory for
   results. Coarse, but it works and keeps you off unstable internals.
3. **Build the C-ABI library yourself.** `gt-ffi` compiles as `staticlib` /
   `cdylib`. You can link it and call exported `extern "C"` functions, but the
   surface is minimal and unstable — expect to extend `gt-ffi` first.

Recommendation: if you can run Rust in your stack, depend on `gt-core` /
`gt-flows` directly (sections 1–5). Otherwise, shell out to a CLI until the
`uniffi` surface lands.

## Current limitations

- The crates are version `0.1.0` and pre-release; the public API is not stable.
- The real backend (`LlamaCppBackend`) is Metal-accelerated and validated on
  Apple Silicon. Other targets are untested.
- `/class-plan` requires a Typst templates directory and the `tesseract` and
  `typst` toolchains available for OCR and PDF rendering.
- There is no plugin registration API yet — adding a new flow means adding a Rust
  module to `gt-flows`. See the repo `README.md` "How to Contribute" section.
