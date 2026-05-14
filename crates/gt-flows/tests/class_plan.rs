//! End-to-end integration test for `/class-plan` with mocked OCR + PDF + backend.

use chrono::NaiveDate;
use gt_core::backend::{MockBackend, MockScript, StopReason};
use gt_core::tool::ToolRegistry;
use gt_flows::class_plan::flow_with_ctx;
use gt_flows::orchestrator::Orchestrator;
use gt_tools::{MockOcrRunner, MockPdfRunner};
use std::sync::Arc;
use tempfile::tempdir;

fn drain_handle(handle: &mut gt_flows::orchestrator::OrchestratorHandle) -> tokio::task::JoinHandle<()> {
    let mut flow_rx = std::mem::replace(&mut handle.flow_events, tokio::sync::mpsc::channel(1).1);
    let session_rxs: Vec<_> = handle.session_events.drain().map(|(_, rx)| rx).collect();
    tokio::spawn(async move {
        let _drains: Vec<_> = session_rxs
            .into_iter()
            .map(|mut rx| tokio::spawn(async move { while rx.recv().await.is_some() {} }))
            .collect();
        while flow_rx.recv().await.is_some() {}
    })
}

#[tokio::test]
async fn class_plan_end_to_end_with_mocks() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();

    // Pre-stage a student (matches /student-add's outputs).
    let maya_dir = root.join("students").join("maya");
    tokio::fs::create_dir_all(&maya_dir).await.unwrap();
    tokio::fs::write(maya_dir.join("student.md"), "# Maya\nlikes anime").await.unwrap();
    tokio::fs::write(maya_dir.join("tags.json"), r#"["anime","drawing"]"#).await.unwrap();

    // Mock a fake PDF input.
    let pdf_path = root.join("source.pdf");
    tokio::fs::write(&pdf_path, b"fake pdf bytes").await.unwrap();

    let ocr = Arc::new(MockOcrRunner {
        text: "Chapter 3 — Photosynthesis. Plants convert light into chemical energy via chloroplasts."
            .into(),
    });
    let pdf = Arc::new(MockPdfRunner);

    let backend = Arc::new(MockBackend::new());
    // -- write-class-notes (Read source.txt -> Write class-notes.md -> Done)
    backend.push(
        MockScript::new()
            .tool("Read", serde_json::json!({"path":"source.txt"}))
            .done(StopReason::Eos),
    );
    backend.push(
        MockScript::new()
            .tool(
                "Write",
                serde_json::json!({
                    "path":"class-notes.md",
                    "content":"# Photosynthesis\n## Learning objectives\n- Identify chloroplasts.\n## Key concepts\n### Light reaction\n- bullet\n## Worked example\n- One.\n## Common misconceptions\n- Plants don't 'eat' soil.\n",
                }),
            )
            .done(StopReason::Eos),
    );
    backend.push(MockScript::new().text("Done.").done(StopReason::Eos));

    // -- write-homework (Read class-notes.md -> Write homework.md -> Done)
    backend.push(
        MockScript::new()
            .tool("Read", serde_json::json!({"path":"class-notes.md"}))
            .done(StopReason::Eos),
    );
    backend.push(
        MockScript::new()
            .tool(
                "Write",
                serde_json::json!({
                    "path":"homework.md",
                    "content":"# Homework — Photosynthesis\n## Practice problems\n1. one\n2. two\n3. three\n4. four\n5. five\n## Reflection prompt\nWhy?\n## Suggested time\n30 minutes\n",
                }),
            )
            .done(StopReason::Eos),
    );
    backend.push(MockScript::new().text("Done.").done(StopReason::Eos));

    // -- tailor-for-maya: Read 4x, Write notes.md, Write homework.md, Done
    for path in ["student.md", "tags.json", "class-notes.md", "homework.md"] {
        backend.push(
            MockScript::new()
                .tool("Read", serde_json::json!({"path": path}))
                .done(StopReason::Eos),
        );
    }
    backend.push(
        MockScript::new()
            .tool(
                "Write",
                serde_json::json!({
                    "path":"notes.md",
                    "content":"# Photosynthesis (Maya — anime edition)\n## Learning objectives\n- ...\n## Key concepts\n### Light reaction\n- via Studio Ghibli analogies\n## Worked example\n- One.\n## Common misconceptions\n- ...\n",
                }),
            )
            .done(StopReason::Eos),
    );
    backend.push(
        MockScript::new()
            .tool(
                "Write",
                serde_json::json!({
                    "path":"homework.md",
                    "content":"# Homework — Photosynthesis\n## Practice problems\n1. drawing-themed\n2. anime-themed\n3. three\n4. four\n5. five\n## Reflection prompt\nWhy?\n## Suggested time\n30 minutes\n",
                }),
            )
            .done(StopReason::Eos),
    );
    backend.push(MockScript::new().text("Done.").done(StopReason::Eos));

    let tools = ToolRegistry::new()
        .register(Arc::new(gt_tools::ReadTool))
        .register(Arc::new(gt_tools::WriteTool))
        .register(Arc::new(gt_tools::EditTool));

    let templates_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap() // crates/
        .parent()
        .unwrap() // repo root
        .join("templates/typst");

    let (flow, ctx) = flow_with_ctx(
        root.clone(),
        NaiveDate::from_ymd_opt(2026, 5, 15).unwrap(),
        pdf_path,
        ocr,
        pdf,
        templates_dir,
    )
    .expect("flow built");

    let orch = Orchestrator::new(backend, tools);
    let mut handle = orch.start(flow, ctx);
    let drain = drain_handle(&mut handle);
    let res = handle.join.await.expect("join");
    drain.await.ok();
    res.expect("flow ok");

    // Validate everything landed on disk.
    let lesson = root.join("lessons/2026-05-15");
    assert!(lesson.join("source.txt").exists());
    assert!(lesson.join("class-notes.md").exists());
    assert!(lesson.join("homework.md").exists());
    assert!(lesson.join("per-student/maya/notes.md").exists());
    assert!(lesson.join("per-student/maya/homework.md").exists());
    // PDFs: MockPdfRunner produces plain-text stubs, but the files should exist.
    assert!(lesson.join("class-notes.pdf").exists());
    assert!(lesson.join("homework.pdf").exists());
    assert!(lesson.join("per-student/maya/notes.pdf").exists());
    assert!(lesson.join("per-student/maya/homework.pdf").exists());
}
