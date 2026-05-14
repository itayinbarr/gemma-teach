//! `uniffi-rs`-based bindings facade for the future iPhone frontend.
//!
//! This is intentionally a small surface: open a session, drain its events,
//! submit user inputs. Phase-2 iOS work will define the chunky FFI shapes for
//! flows and TUI features. Today this exists primarily to:
//!
//!   1. Prove that nothing in `gt-core` / `gt-flows` is FFI-hostile (everything
//!      compiles cleanly under the `uniffi` builder).
//!   2. Give the iOS app a stable entry point name to depend on.

use gt_core::backend::{EchoBackend, LlmBackend, MockBackend};
use gt_core::session::{EventSink, SessionBuilder};
use gt_core::tool::ToolRegistry;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Snapshot of one session event flattened into a JSON string. The iOS layer
/// will deserialize this; treating it as JSON across the FFI boundary avoids
/// chunky `Record` shapes that uniffi would otherwise generate.
#[derive(Debug, Clone)]
pub struct FfiSessionEvent {
    pub json: String,
}

#[derive(Debug, Clone, Copy)]
pub enum FfiBackendKind {
    Mock,
    Echo,
}

/// Phase-2 entry point: spin up a single dummy session against the requested
/// backend and return its event stream as a vector of JSON strings. Once the
/// iPhone app is wired, this becomes an `AsyncStream<FfiSessionEvent>` via
/// `uniffi`'s `#[export]` proc-macros. Kept lightweight for now.
pub async fn run_demo_session(
    backend: FfiBackendKind,
    system_prompt: String,
    task_prompt: String,
) -> Vec<FfiSessionEvent> {
    let backend: Arc<dyn LlmBackend> = match backend {
        FfiBackendKind::Mock => Arc::new(MockBackend::new()),
        FfiBackendKind::Echo => Arc::new(EchoBackend::new()),
    };

    let (sink, mut rx) = EventSink::channel(64);
    let drain: tokio::task::JoinHandle<Vec<FfiSessionEvent>> = tokio::spawn(async move {
        let mut out = Vec::new();
        while let Some(ev) = rx.recv().await {
            out.push(FfiSessionEvent {
                json: serde_json::to_string(&ev).unwrap_or_else(|_| "{}".into()),
            });
        }
        out
    });

    let tools = ToolRegistry::new();
    let _ = SessionBuilder::new("ffi-demo", std::env::temp_dir())
        .system_prompt(system_prompt)
        .task_prompt(task_prompt)
        .allowed_tools(Vec::<String>::new())
        .event_sink(sink)
        .model_profile(gt_core::ModelProfile::test_default())
        .run(backend, tools)
        .await;

    drain.await.unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ffi_demo_session_emits_events() {
        let evs = run_demo_session(FfiBackendKind::Echo, "test".into(), "hello".into()).await;
        assert!(!evs.is_empty());
        // Last event is either Done or Failed; both serialize to JSON with a `kind` field.
        assert!(evs.iter().any(|e| e.json.contains("\"kind\":\"done\"") || e.json.contains("\"kind\":\"failed\"")));
    }
}

/// `_force_send_sync<T>` is a compile-time assertion that the named types are
/// `Send + Sync` (a uniffi prerequisite). This is the iOS portability guard:
/// if any public type in `gt-core` becomes non-Send/Sync, this fails to
/// compile and we catch the regression at CI time.
#[allow(dead_code)]
fn _force_send_sync() {
    fn assert<T: Send + Sync + 'static>() {}
    assert::<gt_core::session_event::SessionEvent>();
    assert::<gt_core::session_event::FlowEvent>();
    assert::<gt_core::ChatMessage>();
    assert::<gt_core::ToolRegistry>();
    assert::<mpsc::Receiver<gt_core::session_event::SessionEvent>>();
}
