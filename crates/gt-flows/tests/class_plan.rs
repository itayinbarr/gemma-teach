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
    // The class-notes pipeline is now decomposed into a plan + per-part
    // sessions assembled deterministically. Each mock script below writes
    // one small file; assemble-class-notes (deterministic) concatenates
    // them into `class-notes.md`.
    let class_notes_plan = "# Class-notes plan\n\n## Title\nPhotosynthesis\n\n## Concepts\n- concept: Light reaction\n- concept: Calvin cycle\n- concept: Chloroplasts\n";
    let concept_1 = "### Light reaction\n- Splits water in the thylakoid membrane.\n- Produces ATP and NADPH.\n- Requires sunlight.\n";
    let concept_2 = "### Calvin cycle\n- Runs in the stroma, doesn't need light directly.\n- Fixes CO2 into G3P using ATP and NADPH.\n- Uses the enzyme RuBisCO.\n";
    let concept_3 = "### Chloroplasts\n- Organelles where photosynthesis takes place.\n- Contain stacks of thylakoids (grana).\n- Surrounded by a double membrane.\n";
    let objectives = "## Learning objectives\n- Identify chloroplasts and label their parts.\n- Explain how the light reaction couples to the Calvin cycle.\n- Apply the photosynthesis equation to a worked example.\n";
    let worked_example = "## Worked example\n- A chloroplast captures 6 photons in the light reaction, producing ATP and NADPH that feed one turn of the Calvin cycle to fix one CO2 into G3P.\n";
    let misconceptions = "## Common misconceptions\n- Plants eat soil rather than producing food from light.\n- The Calvin cycle needs sunlight directly.\n- Only the leaves of a plant photosynthesize.\n";
    // Master homework must include the `(maps to: <Concept>)` suffix on every
    // numbered problem — enforced by the validate-homework-mapping step. The
    // concept name in `(maps to: …)` must match one of the assembled
    // class-notes' `### <concept>` headings.
    let homework_body = "# Homework — Photosynthesis\n## Practice problems\n1. one (maps to: Light reaction)\n2. two (maps to: Calvin cycle)\n3. three (maps to: Chloroplasts)\n4. four (maps to: Light reaction)\n5. five (maps to: Calvin cycle)\n## Reflection prompt\nWhy?\n## Suggested time\n30 minutes\n";

    // Flow steps that exercise the backend, in declaration order:
    //   plan-class-notes
    //   summarize-concept-1
    //   summarize-concept-2
    //   summarize-concept-3
    //   write-class-notes-objectives
    //   write-class-notes-worked-example
    //   write-class-notes-misconceptions
    //   write-homework
    //   plan-tailoring-for-maya
    //   tailor-hw-for-maya
    let tailoring_plan = "# Tailoring plan\n\n## Concepts\n- concept: Light reaction\n  interest: studio-ghibli\n  named_element: the kelp forest in Ponyo\n- concept: Calvin cycle\n  interest: studio-ghibli\n  named_element: Spirited Away's bathhouse garden\n- concept: Chloroplasts\n  interest: studio-ghibli\n  named_element: Totoro's camphor tree\n\n## Worked example\n- interest: studio-ghibli\n- named_element: the camphor tree in My Neighbor Totoro\n\n## Problems\n- n: 1\n  interest: studio-ghibli\n  named_element: Ponyo's underwater kelp\n- n: 2\n  interest: studio-ghibli\n  named_element: Spirited Away bathhouse garden\n- n: 3\n  interest: studio-ghibli\n  named_element: the Ghibli forest\n- n: 4\n  interest: studio-ghibli\n  named_element: the camphor tree in Totoro\n- n: 5\n  interest: studio-ghibli\n  named_element: a Ghibli sunset\n";
    for (path, content) in [
        ("class-notes-plan.md", class_notes_plan),
        ("concept-1.md", concept_1),
        ("concept-2.md", concept_2),
        ("concept-3.md", concept_3),
        ("objectives.md", objectives),
        ("worked-example.md", worked_example),
        ("misconceptions.md", misconceptions),
        ("homework.md", homework_body),
        ("tailoring-plan.md", tailoring_plan),
        (
            "homework.md",
            // Tailored body must diverge enough from the master to clear the
            // validate-tailor-divergence step (30 % of body lines must be new).
            "# Homework — Photosynthesis\n## Practice problems\n1. Maya — explain how Ponyo's underwater plants would handle blue-light filtering. (maps to: Light reaction)\n2. Sketch a leaf in Spirited Away's bathhouse garden and label its chloroplasts. (maps to: Calvin cycle)\n3. Compare a Ghibli forest to a real chloroplast in three short bullets. (maps to: Chloroplasts)\n4. Predict what Totoro's giant camphor tree does at night, in terms of photosynthesis. (maps to: Light reaction)\n5. Identify three pigments other than chlorophyll using the colors in a Ghibli sunset. (maps to: Calvin cycle)\n## Reflection prompt\nIf My Neighbor Totoro shifted to winter, what slows down for the camphor tree?\n## Suggested time\n25 minutes\n",
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
    // The decomposed pipeline writes the per-part files first; the assembler
    // concatenates them into class-notes.md.
    assert!(lesson.join("class-notes-plan.md").exists());
    assert!(lesson.join("concept-1.md").exists());
    assert!(lesson.join("concept-2.md").exists());
    assert!(lesson.join("concept-3.md").exists());
    assert!(lesson.join("objectives.md").exists());
    assert!(lesson.join("worked-example.md").exists());
    assert!(lesson.join("misconceptions.md").exists());
    assert!(lesson.join("class-notes.md").exists());
    let assembled = std::fs::read_to_string(lesson.join("class-notes.md")).unwrap();
    for needle in [
        "# Photosynthesis",
        "## Learning objectives",
        "## Key concepts",
        "### Light reaction",
        "### Calvin cycle",
        "### Chloroplasts",
        "## Worked example",
        "## Common misconceptions",
    ] {
        assert!(
            assembled.contains(needle),
            "assembled class-notes.md missing '{needle}':\n{assembled}"
        );
    }
    assert!(lesson.join("homework.md").exists());
    assert!(lesson.join("per-student/maya/tailoring-plan.md").exists());
    assert!(lesson.join("per-student/maya/homework.md").exists());
    assert!(lesson.join("class-notes.pdf").exists());
    assert!(lesson.join("homework.pdf").exists());
    assert!(lesson.join("per-student/maya/homework.pdf").exists());
}
