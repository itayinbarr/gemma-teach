use gt_core::backend::LlmBackend;
use gt_core::ids::StepId;
use gt_core::session::EventSink;
use gt_core::session_event::{FlowEvent, SessionEvent, StepDescriptor, StepState};
use gt_core::tool::ToolRegistry;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, Semaphore};

use crate::context::FlowCtx;
use crate::step::{Flow, FlowError, StepKind, StepOutcome};

/// Default concurrency cap for parallel-group steps. Conservative — Gemma 4
/// E2B on Metal uses the whole GPU per inference, so the win from concurrency
/// is small and the memory cost is high. Power users can override.
pub const DEFAULT_PARALLELISM: usize = 1;

/// Returned by `Orchestrator::start`. Owns the channels the TUI subscribes to
/// plus a join handle for the orchestrator task.
pub struct OrchestratorHandle {
    pub flow_events: mpsc::Receiver<FlowEvent>,
    pub session_events: HashMap<StepId, mpsc::Receiver<SessionEvent>>,
    pub join: tokio::task::JoinHandle<Result<FlowCtx, FlowError>>,
}

pub struct Orchestrator {
    backend: Arc<dyn LlmBackend>,
    tools: ToolRegistry,
    parallelism: usize,
}

impl Orchestrator {
    pub fn new(backend: Arc<dyn LlmBackend>, tools: ToolRegistry) -> Self {
        Self {
            backend,
            tools,
            parallelism: DEFAULT_PARALLELISM,
        }
    }

    pub fn with_parallelism(mut self, n: usize) -> Self {
        self.parallelism = n.max(1);
        self
    }

    /// Kick off the flow. Returns immediately with channels you can subscribe
    /// to; the run completes asynchronously and the final `FlowCtx` is
    /// available via `handle.join.await`.
    pub fn start(self, flow: Flow, ctx: FlowCtx) -> OrchestratorHandle {
        let (flow_tx, flow_rx) = mpsc::channel::<FlowEvent>(256);

        // Pre-create per-step session channels so subscribers can attach
        // before the orchestrator starts streaming.
        let mut session_channels: HashMap<StepId, mpsc::Receiver<SessionEvent>> = HashMap::new();
        let mut session_senders: HashMap<StepId, mpsc::Sender<SessionEvent>> = HashMap::new();
        for s in &flow.steps {
            if matches!(s.kind, StepKind::Agent(_)) {
                let (tx, rx) = mpsc::channel(256);
                session_senders.insert(s.id, tx);
                session_channels.insert(s.id, rx);
            }
        }

        let step_desc: Vec<StepDescriptor> = flow
            .steps
            .iter()
            .map(|s| StepDescriptor {
                id: s.id,
                name: s.name.clone(),
                kind: match s.kind {
                    StepKind::Deterministic(_) => "deterministic".into(),
                    StepKind::Agent(_) => "agent".into(),
                },
            })
            .collect();

        let backend = self.backend.clone();
        let tools = self.tools.clone();
        let parallelism = self.parallelism;
        let flow_id = flow.id;
        let flow_name = flow.name.clone();
        let steps = flow.steps;

        let join = tokio::spawn(async move {
            let _ = flow_tx
                .send(FlowEvent::FlowStarted {
                    id: flow_id,
                    name: flow_name,
                    steps: step_desc,
                })
                .await;

            let ctx = Arc::new(Mutex::new(ctx));
            let sem = Arc::new(Semaphore::new(parallelism));

            // Walk steps as adjacent runs of (same parallel_group) or singletons.
            // Drain steps so we can move owned StepKinds into tasks.
            let mut steps = steps;
            let mut i = 0usize;
            while i < steps.len() {
                let group = steps[i].parallel_group.clone();
                let mut end = i + 1;
                if group.is_some() {
                    while end < steps.len() && steps[end].parallel_group == group {
                        end += 1;
                    }
                }
                // Drain this chunk of steps. We use `Option` swap to take ownership
                // of `kind` without invalidating the rest of the Vec.
                let mut chunk: Vec<(StepId, String, StepKind)> = Vec::with_capacity(end - i);
                for s in steps[i..end].iter_mut() {
                    let mut placeholder = StepKind::Deterministic(Box::new(NoopDet));
                    std::mem::swap(&mut s.kind, &mut placeholder);
                    chunk.push((s.id, s.name.clone(), placeholder));
                }

                if group.is_some() && chunk.len() > 1 {
                    // Concurrent execution under the semaphore.
                    let mut tasks = Vec::with_capacity(chunk.len());
                    for (step_id, step_name, kind) in chunk.into_iter() {
                        let sem = sem.clone();
                        let backend = backend.clone();
                        let tools = tools.clone();
                        let flow_tx = flow_tx.clone();
                        let session_sender = session_senders.remove(&step_id);
                        let ctx = ctx.clone();
                        tasks.push(tokio::spawn(async move {
                            let _permit = sem.acquire_owned().await.expect("semaphore closed");
                            run_step(
                                step_id,
                                step_name,
                                kind,
                                ctx,
                                backend,
                                tools,
                                flow_tx,
                                session_sender,
                            )
                            .await
                        }));
                    }
                    for t in tasks {
                        match t.await {
                            Ok(Ok(())) => {}
                            Ok(Err(e)) => {
                                let _ = flow_tx
                                    .send(FlowEvent::FlowDone {
                                        id: flow_id,
                                        ok: false,
                                    })
                                    .await;
                                return Err(e);
                            }
                            Err(je) => {
                                let _ = flow_tx
                                    .send(FlowEvent::FlowDone {
                                        id: flow_id,
                                        ok: false,
                                    })
                                    .await;
                                return Err(FlowError::Internal(format!("join: {je}")));
                            }
                        }
                    }
                } else {
                    // Sequential execution — single step or non-grouped run.
                    for (step_id, step_name, kind) in chunk.into_iter() {
                        let session_sender = session_senders.remove(&step_id);
                        if let Err(e) = run_step(
                            step_id,
                            step_name,
                            kind,
                            ctx.clone(),
                            backend.clone(),
                            tools.clone(),
                            flow_tx.clone(),
                            session_sender,
                        )
                        .await
                        {
                            let _ = flow_tx
                                .send(FlowEvent::FlowDone {
                                    id: flow_id,
                                    ok: false,
                                })
                                .await;
                            return Err(e);
                        }
                    }
                }
                i = end;
            }

            let _ = flow_tx
                .send(FlowEvent::FlowDone {
                    id: flow_id,
                    ok: true,
                })
                .await;

            let final_ctx = Arc::try_unwrap(ctx)
                .map_err(|_| FlowError::Internal("ctx still shared at flow end".into()))?
                .into_inner();
            Ok(final_ctx)
        });

        OrchestratorHandle {
            flow_events: flow_rx,
            session_events: session_channels,
            join,
        }
    }
}

async fn run_step(
    step_id: StepId,
    step_name: String,
    kind: StepKind,
    ctx: Arc<Mutex<FlowCtx>>,
    backend: Arc<dyn LlmBackend>,
    tools: ToolRegistry,
    flow_tx: mpsc::Sender<FlowEvent>,
    session_sender: Option<mpsc::Sender<SessionEvent>>,
) -> Result<(), FlowError> {
    let _ = flow_tx
        .send(FlowEvent::StepStateChanged {
            step: step_id,
            state: StepState::Running,
        })
        .await;

    let outcome = match kind {
        StepKind::Deterministic(s) => {
            let snapshot = ctx.lock().await.clone();
            s.run(&snapshot).await.map_err(|e| {
                tracing::error!(step = %step_name, "deterministic step failed: {e}");
                e
            })?
        }
        StepKind::Agent(factory) => {
            let _ = flow_tx
                .send(FlowEvent::StepStateChanged {
                    step: step_id,
                    state: StepState::Streaming,
                })
                .await;
            // Build the session against a snapshot of the context.
            let mut sb = {
                let snap = ctx.lock().await;
                factory.build(&snap)
            };
            // Wire the orchestrator-provided sink so SessionEvents reach the TUI.
            if let Some(tx) = session_sender {
                let (sink, mut rx) = EventSink::channel(256);
                sb = sb.event_sink(sink);
                let forward_tx = tx.clone();
                tokio::spawn(async move {
                    while let Some(ev) = rx.recv().await {
                        if forward_tx.send(ev).await.is_err() {
                            break;
                        }
                    }
                });
            }
            sb.run(backend.clone(), tools.clone()).await.map_err(|e| {
                FlowError::Step {
                    step: step_name.clone(),
                    msg: e.to_string(),
                }
            })?;
            // Outputs are declared by the factory and become artifacts.
            let outs = factory.output_keys();
            StepOutcome { outputs: outs }
        }
    };

    // Merge produced artifacts.
    {
        let mut guard = ctx.lock().await;
        for (k, path) in &outcome.outputs {
            guard.artifacts.insert(k.clone(), path.clone());
            let _ = flow_tx
                .send(FlowEvent::StepArtifactProduced {
                    step: step_id,
                    key: k.clone(),
                    path: path.display().to_string(),
                })
                .await;
        }
    }
    let _ = flow_tx
        .send(FlowEvent::StepStateChanged {
            step: step_id,
            state: StepState::Done,
        })
        .await;
    Ok(())
}

// Placeholder needed to take ownership of `StepKind` out of a `&mut StepNode`
// via `std::mem::swap`. Never actually runs — we always immediately consume
// the original kind.
struct NoopDet;
#[async_trait::async_trait]
impl crate::step::DeterministicStep for NoopDet {
    async fn run(&self, _ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        Ok(StepOutcome::default())
    }
}

impl Clone for FlowCtx {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            date: self.date,
            inputs: self.inputs.clone(),
            artifacts: self.artifacts.clone(),
        }
    }
}
