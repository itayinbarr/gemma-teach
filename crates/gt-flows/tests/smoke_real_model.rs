//! Real-model smoke test for `/student-add`. Gated on `GEMMA_TEACH_SMOKE=1`
//! because it requires the ~3.5 GB GGUF on disk and Metal-capable hardware.
//! Skipped silently otherwise so CI stays fast.

#![cfg(feature = "smoke")]

use chrono::NaiveDate;
use gt_core::llama_backend::{LlamaCppBackend, LlamaConfig};
use gt_core::model_fetch::default_models_dir;
use gt_core::tool::ToolRegistry;
use gt_flows::orchestrator::Orchestrator;
use gt_flows::student_add::flow_with_ctx;
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn student_add_against_real_gemma() {
    if std::env::var("GEMMA_TEACH_SMOKE").ok().as_deref() != Some("1") {
        eprintln!("skipping: set GEMMA_TEACH_SMOKE=1 to run this against the real model");
        return;
    }
    let model_path = std::env::var("GEMMA_TEACH_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| default_models_dir().join("gemma-3n-E2B-it-Q4_K_M.gguf"));
    assert!(
        model_path.exists(),
        "set GEMMA_TEACH_MODEL or run `gemma-teach --download-only` first"
    );

    let dir = tempdir().unwrap();
    let backend = Arc::new(LlamaCppBackend::new(LlamaConfig::new(model_path)));
    let tools = ToolRegistry::new()
        .register(Arc::new(gt_tools::ReadTool))
        .register(Arc::new(gt_tools::WriteTool))
        .register(Arc::new(gt_tools::EditTool));
    let (flow, ctx) = flow_with_ctx(
        dir.path().to_path_buf(),
        NaiveDate::from_ymd_opt(2026, 5, 15).unwrap(),
        "Maya".into(),
        "Twelve years old. Loves Studio Ghibli films and marine biology. Plays piano.".into(),
    );
    let orch = Orchestrator::new(backend, tools);
    let mut handle = orch.start(flow, ctx);
    let flow_drain = tokio::spawn(async move {
        while handle.flow_events.recv().await.is_some() {}
    });
    for (_id, mut rx) in handle.session_events.drain() {
        tokio::spawn(async move { while rx.recv().await.is_some() {} });
    }
    let _ = handle.join.await.unwrap().expect("flow ok against real model");
    flow_drain.await.ok();

    let md = dir.path().join("students/maya/student.md");
    let tags = dir.path().join("students/maya/tags.json");
    assert!(md.exists(), "student.md should exist");
    assert!(tags.exists(), "tags.json should exist");
    let parsed: Vec<String> =
        serde_json::from_str(&tokio::fs::read_to_string(&tags).await.unwrap()).unwrap();
    assert!(!parsed.is_empty(), "tags array should be non-empty");
}
