//! End-to-end integration test for `/class-plan` with mocked OCR + PDF + backend.

use chrono::NaiveDate;
use gt_core::backend::{MockBackend, MockScript, StopReason};
use gt_core::tool::ToolRegistry;
use gt_flows::class_plan::flow_with_ctx;
use gt_flows::orchestrator::Orchestrator;
use gt_tools::{MockOcrRunner, MockPdfRunner};
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn class_plan_end_to_end_with_mocks() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();

    // Pre-stage a student so /class-plan has someone to tailor for.
    let maya_dir = root.join("students").join("maya");
    tokio::fs::create_dir_all(&maya_dir).await.unwrap();
    tokio::fs::write(maya_dir.join("student.md"), "# Maya\nlikes anime").await.unwrap();
    tokio::fs::write(maya_dir.join("tags.json"), r#"["anime","drawing"]"#).await.unwrap();

    let pdf_path = root.join("source.pdf");
    tokio::fs::write(&pdf_path, b"fake pdf bytes").await.unwrap();

    let ocr = Arc::new(MockOcrRunner {
        text: "Chapter 3 — Photosynthesis. Plants convert light into chemical energy via chloroplasts."
            .into(),
    });
    let pdf = Arc::new(MockPdfRunner);

    let backend = Arc::new(MockBackend::new());
    let class_notes_body = "# Photosynthesis\n## Learning objectives\n- Identify chloroplasts.\n## Key concepts\n### Light reaction\n- bullet\n## Worked example\n- One.\n## Common misconceptions\n- Plants don't 'eat' soil.\n";
    let homework_body = "# Homework — Photosynthesis\n## Practice problems\n1. one\n2. two\n3. three\n4. four\n5. five\n## Reflection prompt\nWhy?\n## Suggested time\n30 minutes\n";

    // Flow steps that exercise the backend (in order):
    //   write-class-notes    (one-shot Write)
    //   write-homework       (one-shot Write)
    //   tailor-notes-for-maya  (one-shot Write notes.md)
    //   tailor-hw-for-maya     (one-shot Write homework.md)
    for (path, content) in [
        ("class-notes.md", class_notes_body),
        ("homework.md", homework_body),
        (
            "notes.md",
            "# Photosynthesis (Maya — anime edition)\n## Learning objectives\n- ...\n## Key concepts\n### Light reaction\n- via Studio Ghibli analogies\n## Worked example\n- One.\n## Common misconceptions\n- ...\n",
        ),
        (
            "homework.md",
            "# Homework — Photosynthesis\n## Practice problems\n1. drawing\n2. anime\n3. three\n4. four\n5. five\n## Reflection prompt\nWhy?\n## Suggested time\n30 minutes\n",
        ),
    ] {
        backend.push(
            MockScript::new()
                .text("Done.")
                .tool(
                    "Write",
                    serde_json::json!({ "path": path, "content": content }),
                )
                .done(StopReason::Eos),
        );
    }

    let tools = ToolRegistry::new()
        .register(Arc::new(gt_tools::ReadTool))
        .register(Arc::new(gt_tools::WriteTool))
        .register(Arc::new(gt_tools::EditTool));

    let templates_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
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
    let flow_drain = tokio::spawn(async move {
        while handle.flow_events.recv().await.is_some() {}
    });
    for (_id, mut rx) in handle.session_events.drain() {
        tokio::spawn(async move {
            while rx.recv().await.is_some() {}
        });
    }
    let res = handle.join.await.expect("join");
    flow_drain.await.ok();
    res.expect("flow ok");

    let lesson = root.join("lessons/2026-05-15");
    assert!(lesson.join("source.txt").exists());
    assert!(lesson.join("class-notes.md").exists());
    assert!(lesson.join("homework.md").exists());
    assert!(lesson.join("per-student/maya/notes.md").exists());
    assert!(lesson.join("per-student/maya/homework.md").exists());
    assert!(lesson.join("class-notes.pdf").exists());
    assert!(lesson.join("homework.pdf").exists());
    assert!(lesson.join("per-student/maya/notes.pdf").exists());
    assert!(lesson.join("per-student/maya/homework.pdf").exists());
}
