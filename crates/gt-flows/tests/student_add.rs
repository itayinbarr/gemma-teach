//! End-to-end integration test for `/student-add` against a scripted MockBackend.

use chrono::NaiveDate;
use gt_core::backend::{MockBackend, MockScript, StopReason};
use gt_core::tool::ToolRegistry;
use gt_flows::orchestrator::Orchestrator;
use gt_flows::student_add::flow_with_ctx;
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn student_add_end_to_end_against_mock_backend() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();

    let student_md_content = "# Maya\n\n## Snapshot\n- 12 years old\n- loves storytelling\n\n## Interests\n- Studio Ghibli films\n- marine biology\n\n## Hobbies\n- piano\n- drawing\n\n## Media they love\n- Spirited Away\n- The Owl House\n\n## Notes for tailoring lessons\n- Visual learner — sketches help\n- Frame examples around animals and stories\n";

    let backend = Arc::new(MockBackend::new());

    // -- session: write-student
    backend.push(
        MockScript::new()
            .tool(
                "Write",
                serde_json::json!({
                    "path": "student.md",
                    "content": student_md_content,
                }),
            )
            .done(StopReason::Eos),
    );
    backend.push(MockScript::new().text("Done.").done(StopReason::Eos));

    // -- session: extract-tags (one-shot: student.md is pre-loaded into the prompt)
    backend.push(
        MockScript::new()
            .text("Done.")
            .tool(
                "Write",
                serde_json::json!({
                    "path": "tags.json",
                    "content": "[\"studio-ghibli\",\"marine-biology\",\"piano\",\"drawing\",\"storytelling\"]",
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
        "Maya is twelve, loves Studio Ghibli films and marine biology. Plays piano and draws. Watches The Owl House.".into(),
    );

    let orch = Orchestrator::new(backend, tools);
    let mut handle = orch.start(flow, ctx);

    // Drain events so the channels don't backpressure the orchestrator.
    let flow_drain = tokio::spawn(async move {
        while handle.flow_events.recv().await.is_some() {}
    });
    for (_id, mut rx) in handle.session_events.drain() {
        tokio::spawn(async move {
            while rx.recv().await.is_some() {}
        });
    }

    let final_ctx = handle
        .join
        .await
        .expect("join")
        .expect("flow ok");
    flow_drain.await.unwrap();

    let student_dir = root.join("students").join("maya");
    let student_md = student_dir.join("student.md");
    let tags_json = student_dir.join("tags.json");
    let intersections = student_dir.join("intersections.json");

    assert!(student_md.exists(), "student.md should be written");
    assert!(tags_json.exists(), "tags.json should be written");
    assert!(intersections.exists(), "intersections.json should be written");

    let got_md = tokio::fs::read_to_string(&student_md).await.unwrap();
    assert!(got_md.starts_with("# Maya"));

    let got_tags: Vec<String> =
        serde_json::from_str(&tokio::fs::read_to_string(&tags_json).await.unwrap()).unwrap();
    assert!(got_tags.contains(&"studio-ghibli".to_string()));
    assert!(got_tags.contains(&"marine-biology".to_string()));

    // No prior students → empty intersections.
    let inter: serde_json::Value =
        serde_json::from_str(&tokio::fs::read_to_string(&intersections).await.unwrap()).unwrap();
    assert!(inter.is_array());
    assert_eq!(inter.as_array().unwrap().len(), 0);

    let _ = final_ctx;
}

#[tokio::test]
async fn student_add_tag_intersections_finds_overlaps_with_existing_students() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();

    // Pre-stage another student so the intersection step finds an overlap.
    let jonah_dir = root.join("students").join("jonah");
    tokio::fs::create_dir_all(&jonah_dir).await.unwrap();
    tokio::fs::write(jonah_dir.join("student.md"), "# Jonah").await.unwrap();
    tokio::fs::write(
        jonah_dir.join("tags.json"),
        r#"["studio-ghibli","football"]"#,
    )
    .await
    .unwrap();

    let backend = Arc::new(MockBackend::new());
    // write-student
    backend.push(
        MockScript::new()
            .tool(
                "Write",
                serde_json::json!({"path":"student.md","content":"# Maya\n## Interests\n- Studio Ghibli\n"}),
            )
            .done(StopReason::Eos),
    );
    backend.push(MockScript::new().text("Done.").done(StopReason::Eos));
    // extract-tags (one-shot)
    backend.push(
        MockScript::new()
            .text("Done.")
            .tool(
                "Write",
                serde_json::json!({"path":"tags.json","content":"[\"studio-ghibli\",\"drawing\"]"}),
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
        "loves Studio Ghibli and drawing".into(),
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
    flow_drain.await.unwrap();

    let inter_text = tokio::fs::read_to_string(root.join("students/maya/intersections.json"))
        .await
        .unwrap();
    let inter: serde_json::Value = serde_json::from_str(&inter_text).unwrap();
    let arr = inter.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0][0], "jonah");
    assert_eq!(arr[0][1][0], "studio-ghibli");
}
