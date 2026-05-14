//! End-to-end test for `/student-edit` against MockBackend.

use chrono::NaiveDate;
use gt_core::backend::{MockBackend, MockScript, StopReason};
use gt_core::tool::ToolRegistry;
use gt_flows::orchestrator::Orchestrator;
use gt_flows::student_edit::flow_with_ctx;
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn student_edit_rewrites_profile_and_refreshes_tags() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();

    // Pre-stage Maya.
    let maya_dir = root.join("students").join("maya");
    tokio::fs::create_dir_all(&maya_dir).await.unwrap();
    let original = "# Maya\n\n## Interests\n- painting\n- swimming\n";
    tokio::fs::write(maya_dir.join("student.md"), original).await.unwrap();
    tokio::fs::write(maya_dir.join("tags.json"), r#"["painting","swimming"]"#).await.unwrap();

    let backend = Arc::new(MockBackend::new());

    // -- rewrite-student session: one-shot Write of updated profile.
    backend.push(
        MockScript::new()
            .text("Done.")
            .tool(
                "Write",
                serde_json::json!({
                    "path": "student.md",
                    "content": "# Maya\n\n## Interests\n- painting\n- swimming\n- chess (newly picked up)\n",
                }),
            )
            .done(StopReason::Eos),
    );
    // -- refresh-tags session: one-shot Write of tags.json.
    backend.push(
        MockScript::new()
            .text("Done.")
            .tool(
                "Write",
                serde_json::json!({
                    "path": "tags.json",
                    "content": "[\"painting\",\"swimming\",\"chess\"]",
                }),
            )
            .done(StopReason::Eos),
    );

    let tools = ToolRegistry::new()
        .register(Arc::new(gt_tools::ReadTool))
        .register(Arc::new(gt_tools::WriteTool))
        .register(Arc::new(gt_tools::EditTool));

    let (flow, ctx) = flow_with_ctx(
        root.clone(),
        NaiveDate::from_ymd_opt(2026, 5, 15).unwrap(),
        "Maya".into(),
        "Started competitive chess this term. Keep painting and swimming.".into(),
    );

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
    let _ = handle.join.await.expect("join").expect("flow ok");
    flow_drain.await.ok();

    let got_md = tokio::fs::read_to_string(maya_dir.join("student.md")).await.unwrap();
    assert!(got_md.contains("- chess (newly picked up)"));
    assert!(got_md.contains("- painting"));

    let got_tags: Vec<String> =
        serde_json::from_str(&tokio::fs::read_to_string(maya_dir.join("tags.json")).await.unwrap())
            .unwrap();
    assert!(got_tags.contains(&"chess".to_string()));
    // Backup should have been cleaned up.
    assert!(!maya_dir.join(".student.md.prior").exists());
}
