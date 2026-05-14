//! Run `/student-add` against the real Gemma 3n E2B model and dump every
//! event (flow + session) as JSONL. Used for the trace-driven tuning loop
//! described in the plan.
//!
//!     cargo run -p gt-flows --example record_trace --features smoke -- \
//!         --name "Maya" \
//!         --description "12 years old. Loves Studio Ghibli films and marine biology." \
//!         --out traces/student-add-1.jsonl
//!
//! The example writes one JSON object per line. The trailing newline is
//! deliberate; tail -f the file to watch live.

use chrono::NaiveDate;
use gt_core::backend::LlmBackend;
use gt_core::tool::ToolRegistry;
use gt_flows::orchestrator::Orchestrator;
use gt_flows::student_add::flow_with_ctx;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;

#[derive(serde::Parser, Debug)]
#[cfg(any())] // placeholder so cargo-fmt doesn't complain about unused derive — we parse args manually below.
struct Unused;

#[derive(Debug, Default)]
struct Args {
    name: Option<String>,
    description: Option<String>,
    notebook: Option<PathBuf>,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--name" => a.name = it.next(),
            "--description" => a.description = it.next(),
            "--notebook" => a.notebook = it.next().map(PathBuf::from),
            "--out" => a.out = it.next().map(PathBuf::from),
            "--help" | "-h" => {
                eprintln!("record_trace --name <NAME> --description <TEXT> [--notebook <DIR>] [--out <FILE>]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(2);
            }
        }
    }
    a
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
    let name = args.name.unwrap_or_else(|| "Maya".to_string());
    let description = args.description.unwrap_or_else(|| {
        "12 years old. Loves Studio Ghibli films and marine biology. Plays piano and draws.".into()
    });
    let notebook = args.notebook.unwrap_or_else(|| {
        let tmp = tempfile::tempdir().expect("tempdir").keep();
        tmp
    });
    let out = args
        .out
        .unwrap_or_else(|| PathBuf::from("traces/student-add-trace.jsonl"));

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

    let (flow, ctx) = flow_with_ctx(
        notebook.clone(),
        NaiveDate::from_ymd_opt(2026, 5, 15).unwrap(),
        name,
        description,
    );

    let orch = Orchestrator::new(backend, tools);
    let mut handle = orch.start(flow, ctx);

    // Pick up each session-event receiver and route it to the trace file via
    // a tagged record. We need to know which step each event belongs to, so
    // pre-build a map from StepId to step name as we see FlowStarted.
    let mut step_names: std::collections::HashMap<gt_core::ids::StepId, String> =
        std::collections::HashMap::new();

    // Forward all session channels onto a single sink so we can interleave
    // them with flow events in source order.
    let (sink, mut sink_rx) =
        tokio::sync::mpsc::channel::<(String, gt_core::session_event::SessionEvent)>(256);
    let session_rxs: Vec<(gt_core::ids::StepId, _)> = handle.session_events.drain().collect();
    for (id, mut rx) in session_rxs {
        let sink = sink.clone();
        // We need the step name. Until FlowStarted arrives we don't know it,
        // so use the StepId's short hex as a placeholder; we'll rewrite once
        // we see FlowStarted in the join task below.
        let placeholder = format!("step:{id}");
        tokio::spawn(async move {
            while let Some(ev) = rx.recv().await {
                if sink.send((placeholder.clone(), ev)).await.is_err() {
                    break;
                }
            }
        });
    }
    drop(sink); // close once all session tasks are done

    // Per-step counters for tok/s telemetry.
    let mut step_token_count: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();
    let mut step_turn_start: std::collections::HashMap<String, std::time::Instant> =
        std::collections::HashMap::new();
    let total_start = std::time::Instant::now();
    let mut total_tokens: u64 = 0;

    // Drain interleaved.
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
                        // tok/s accounting
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

    // Wait for the orchestrator task to finish (it may have already).
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

#[cfg(feature = "smoke")]
fn build_real_backend() -> Result<Arc<dyn LlmBackend>, Box<dyn std::error::Error>> {
    use gt_core::llama_backend::{LlamaCppBackend, LlamaConfig};
    use gt_core::model_fetch::default_models_dir;
    let path = std::env::var("GEMMA_TEACH_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_models_dir().join("gemma-3n-E2B-it-Q4_K_M.gguf"));
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
