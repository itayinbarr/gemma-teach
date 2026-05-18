//! Drive any of the three feature flows against the real Gemma 4 E2B
//! backend and dump every event as JSONL with per-step tok/s telemetry.
//!
//! Examples
//!
//!     cargo run -p gt-flows --example record_trace --features smoke -- \
//!         student-add --name "Maya" \
//!         --description "12 years old. Loves Studio Ghibli, marine biology." \
//!         --notebook ~/GemmaTeach \
//!         --out traces/maya.jsonl
//!
//!     cargo run -p gt-flows --example record_trace --features smoke -- \
//!         class-plan --source-txt /tmp/chapter.txt \
//!         --notebook ~/GemmaTeach \
//!         --out traces/class-plan.jsonl
//!
//!     cargo run -p gt-flows --example record_trace --features smoke -- \
//!         student-edit --name "Maya" \
//!         --notes "Started chess. Drop swimming." \
//!         --notebook ~/GemmaTeach \
//!         --out traces/maya-edit.jsonl
//!
//! `--notebook` defaults to a fresh tempdir if omitted. `--out` defaults to
//! `traces/<flow>.jsonl`.

use gt_core::backend::LlmBackend;
use gt_core::tool::ToolRegistry;
use gt_flows::orchestrator::Orchestrator;
use gt_flows::Flow;
use gt_flows::FlowCtx;
use gt_tools::{MockOcrRunner, TypstRunner};
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

#[derive(Debug)]
enum Cmd {
    StudentAdd {
        name: String,
        description: String,
    },
    ClassPlan {
        source_txt: PathBuf,
    },
    StudentEdit {
        name: String,
        notes: String,
    },
}

#[derive(Debug, Default)]
struct Args {
    cmd: Option<Cmd>,
    notebook: Option<PathBuf>,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1).peekable();
    let head = match it.next() {
        Some(h) if !h.starts_with("--") => h,
        Some(other) => {
            // Backward-compat: if first arg starts with `--`, default to student-add.
            let mut rest: Vec<String> = vec![other];
            rest.extend(it);
            return reparse_legacy(rest);
        }
        None => {
            print_help();
            std::process::exit(0);
        }
    };

    let mut name: Option<String> = None;
    let mut description: Option<String> = None;
    let mut notes: Option<String> = None;
    let mut source_txt: Option<PathBuf> = None;

    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--name" => name = it.next(),
            "--description" => description = it.next(),
            "--notes" => notes = it.next(),
            "--source-txt" => source_txt = it.next().map(PathBuf::from),
            "--notebook" => a.notebook = it.next().map(PathBuf::from),
            "--out" => a.out = it.next().map(PathBuf::from),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }

    a.cmd = Some(match head.as_str() {
        "student-add" => Cmd::StudentAdd {
            name: name.unwrap_or_else(|| "Maya".into()),
            description: description.unwrap_or_else(|| "12 years old.".into()),
        },
        "class-plan" => Cmd::ClassPlan {
            source_txt: source_txt.expect("class-plan requires --source-txt"),
        },
        "student-edit" => Cmd::StudentEdit {
            name: name.expect("student-edit requires --name"),
            notes: notes.expect("student-edit requires --notes"),
        },
        other => {
            eprintln!("unknown flow: {other}");
            std::process::exit(2);
        }
    });
    a
}

fn reparse_legacy(rest: Vec<String>) -> Args {
    let mut a = Args::default();
    let mut name = None;
    let mut description = None;
    let mut it = rest.into_iter();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--name" => name = it.next(),
            "--description" => description = it.next(),
            "--notebook" => a.notebook = it.next().map(PathBuf::from),
            "--out" => a.out = it.next().map(PathBuf::from),
            _ => {}
        }
    }
    a.cmd = Some(Cmd::StudentAdd {
        name: name.unwrap_or_else(|| "Maya".into()),
        description: description.unwrap_or_else(|| "12 years old.".into()),
    });
    a
}

fn print_help() {
    eprintln!("usage:");
    eprintln!("  record_trace student-add  --name <NAME> --description <TEXT>");
    eprintln!("  record_trace class-plan   --source-txt <FILE>");
    eprintln!("  record_trace student-edit --name <NAME> --notes <TEXT>");
    eprintln!("common: --notebook <DIR>   --out <FILE>");
}

#[derive(Debug, Serialize)]
#[serde(tag = "channel", rename_all = "snake_case")]
enum TraceRecord {
    Flow(gt_core::session_event::FlowEvent),
    Session {
        step: String,
        event: gt_core::session_event::SessionEvent,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let cmd = args.cmd.unwrap();
    let notebook = args.notebook.clone().unwrap_or_else(|| {
        tempfile::tempdir().expect("tempdir").keep()
    });
    tokio::fs::create_dir_all(&notebook).await?;
    let out = args.out.clone().unwrap_or_else(|| match &cmd {
        Cmd::StudentAdd { .. } => "traces/student-add.jsonl".into(),
        Cmd::ClassPlan { .. } => "traces/class-plan.jsonl".into(),
        Cmd::StudentEdit { .. } => "traces/student-edit.jsonl".into(),
    });
    if let Some(parent) = out.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let mut file = tokio::fs::File::create(&out).await?;
    eprintln!("notebook: {}", notebook.display());
    eprintln!("output:   {}", out.display());

    let backend = build_real_backend()?;

    let tools = ToolRegistry::new()
        .register(Arc::new(gt_tools::ReadTool))
        .register(Arc::new(gt_tools::WriteTool))
        .register(Arc::new(gt_tools::EditTool));

    let date = chrono::Local::now().date_naive();
    let templates = templates_dir();
    let (flow, ctx): (Flow, FlowCtx) = match cmd {
        Cmd::StudentAdd { name, description } => {
            gt_flows::student_add::flow_with_ctx(notebook.clone(), date, name, description)
        }
        Cmd::ClassPlan { source_txt } => {
            let source = tokio::fs::read_to_string(&source_txt).await?;
            let ocr: Arc<dyn gt_tools::OcrRunner> = Arc::new(MockOcrRunner { text: source });
            let pdf: Arc<dyn gt_tools::PdfRunner> = Arc::new(TypstRunner::new());
            // /class-plan needs a real pdf path argument even though we mock OCR.
            let dummy_pdf = notebook.join(".dummy.pdf");
            if !dummy_pdf.exists() {
                tokio::fs::write(&dummy_pdf, b"placeholder").await?;
            }
            gt_flows::class_plan::flow_with_ctx(notebook.clone(), date, dummy_pdf, ocr, pdf, templates)?
        }
        Cmd::StudentEdit { name, notes } => {
            gt_flows::student_edit::flow_with_ctx(notebook.clone(), date, name, notes)
        }
    };

    let orch = Orchestrator::new(backend, tools);
    let mut handle = orch.start(flow, ctx);

    let mut step_names: std::collections::HashMap<gt_core::ids::StepId, String> =
        std::collections::HashMap::new();
    let (sink, mut sink_rx) =
        tokio::sync::mpsc::channel::<(String, gt_core::session_event::SessionEvent)>(256);
    let session_rxs: Vec<(gt_core::ids::StepId, _)> = handle.session_events.drain().collect();
    for (id, mut rx) in session_rxs {
        let sink = sink.clone();
        let placeholder = format!("step:{id}");
        tokio::spawn(async move {
            while let Some(ev) = rx.recv().await {
                if sink.send((placeholder.clone(), ev)).await.is_err() {
                    break;
                }
            }
        });
    }
    drop(sink);

    let mut step_token_count: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();
    let mut step_turn_start: std::collections::HashMap<String, std::time::Instant> =
        std::collections::HashMap::new();
    let total_start = std::time::Instant::now();
    let mut total_tokens: u64 = 0;

    loop {
        tokio::select! {
            biased;
            flow_ev = handle.flow_events.recv() => {
                match flow_ev {
                    Some(ev) => {
                        if let gt_core::session_event::FlowEvent::FlowStarted { steps, .. } = &ev {
                            for s in steps {
                                step_names.insert(s.id, s.name.clone());
                            }
                        }
                        emit(&mut file, &TraceRecord::Flow(ev)).await?;
                    }
                    None => break,
                }
            }
            session_ev = sink_rx.recv() => {
                match session_ev {
                    Some((mut step, ev)) => {
                        if let Some(name) = step
                            .strip_prefix("step:")
                            .and_then(|short| {
                                step_names
                                    .iter()
                                    .find(|(id, _)| id.to_string() == short)
                                    .map(|(_, n)| n.clone())
                            })
                        {
                            step = name;
                        }
                        use gt_core::session_event::SessionEvent as SE;
                        match &ev {
                            SE::TurnStarted { turn } => {
                                step_turn_start.insert(step.clone(), std::time::Instant::now());
                                eprintln!("[{step}] turn {turn} started");
                            }
                            SE::TokenDelta { .. } => {
                                *step_token_count.entry(step.clone()).or_insert(0) += 1;
                                total_tokens += 1;
                            }
                            SE::Done { outcome } => {
                                let elapsed = step_turn_start
                                    .remove(&step)
                                    .map(|t| t.elapsed())
                                    .unwrap_or_default();
                                let toks = step_token_count.remove(&step).unwrap_or(0);
                                let tps = if elapsed.as_secs_f64() > 0.0 {
                                    toks as f64 / elapsed.as_secs_f64()
                                } else {
                                    0.0
                                };
                                eprintln!(
                                    "[{step}] done — {toks} tok / {:.2}s = {tps:.2} tok/s (turns: {})",
                                    elapsed.as_secs_f64(),
                                    outcome.turns
                                );
                            }
                            SE::Failed { error, .. } => {
                                eprintln!("[{step}] FAILED — {error}");
                            }
                            _ => {}
                        }
                        emit(&mut file, &TraceRecord::Session { step, event: ev }).await?;
                    }
                    None => {}
                }
            }
        }
    }
    let total_elapsed = total_start.elapsed().as_secs_f64();
    if total_elapsed > 0.0 {
        eprintln!(
            "TOTAL: {total_tokens} tok / {total_elapsed:.2}s = {:.2} tok/s",
            total_tokens as f64 / total_elapsed
        );
    }

    match handle.join.await {
        Ok(Ok(_)) => eprintln!("flow done."),
        Ok(Err(e)) => eprintln!("flow failed: {e}"),
        Err(e) => eprintln!("orchestrator panic: {e}"),
    }
    file.flush().await?;
    eprintln!("trace written to {}", out.display());
    Ok(())
}

async fn emit(file: &mut tokio::fs::File, rec: &TraceRecord) -> std::io::Result<()> {
    let mut line = serde_json::to_vec(rec).map_err(|e| std::io::Error::other(e.to_string()))?;
    line.push(b'\n');
    file.write_all(&line).await?;
    file.flush().await
}

fn templates_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("templates/typst")
}

#[cfg(feature = "smoke")]
fn build_real_backend() -> Result<Arc<dyn LlmBackend>, Box<dyn std::error::Error>> {
    use gt_core::llama_backend::{LlamaCppBackend, LlamaConfig};
    use gt_core::model_fetch::default_models_dir;
    let path = std::env::var("GEMMA_TEACH_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_models_dir().join("gemma-4-E2B-it-Q4_K_M.gguf"));
    if !path.exists() {
        return Err(format!("model not found at {}", path.display()).into());
    }
    eprintln!("loading model: {}", path.display());
    Ok(Arc::new(LlamaCppBackend::new(LlamaConfig::new(path))))
}

#[cfg(not(feature = "smoke"))]
fn build_real_backend() -> Result<Arc<dyn LlmBackend>, Box<dyn std::error::Error>> {
    Err("build with `--features smoke` to enable the real Gemma backend".into())
}
