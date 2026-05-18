//! Focused tests for the validate-homework-mapping deterministic step.
//!
//! The validator enforces the prompt contract: every numbered problem in the
//! homework file must end with ` (maps to: <Concept Name>)`. We exercise it
//! via the same `flow_with_ctx_from_source` path used by `/class-plan`,
//! short-circuiting model work with a MockBackend that emits the homework
//! content we want to validate.
//!
//! Each test stages a hand-crafted `homework.md` (good or bad), then drives
//! the flow up to and including the validator step. We do NOT bother with the
//! tailor sessions or compile-pdfs here — student_slugs is empty so the flow
//! terminates after `validate-homework-mapping`. The earlier validator
//! `validate-tags-json` is fine because we never hit it.

use chrono::NaiveDate;
use gt_core::backend::{MockBackend, MockScript, StopReason};
use gt_core::tool::ToolRegistry;
use gt_flows::class_plan::{flow_with_ctx_from_source, ClassPlanSource};
use gt_flows::orchestrator::Orchestrator;
use gt_tools::{MockOcrRunner, MockPdfRunner};
use std::sync::Arc;
use tempfile::tempdir;

async fn run_with_homework(homework_md: &str) -> Result<(), gt_flows::step::FlowError> {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();

    // No students staged — flow has no tailor steps, so the run ends after
    // validate-homework-mapping (and then compile-pdfs, which uses the mock).
    let ocr = Arc::new(MockOcrRunner {
        text: "Chapter — Photosynthesis. Plants split water using chlorophyll.".into(),
    });
    let pdf = Arc::new(MockPdfRunner);

    let backend = Arc::new(MockBackend::new());
    // The decomposed class-notes pipeline expects one mock script per part.
    // The deterministic assembler then writes `class-notes.md`, so the
    // validator that follows still has a real class-notes file to read
    // its `### <concept>` headings from.
    let plan = "## Title\nPhotosynthesis\n\n## Concepts\n- concept: Light reaction\n- concept: Calvin cycle\n- concept: Chloroplasts\n";
    let concept_1 = "### Light reaction\n- Splits water.\n- Produces ATP and NADPH.\n";
    let concept_2 = "### Calvin cycle\n- Fixes CO2 into G3P.\n- Uses RuBisCO.\n";
    let concept_3 = "### Chloroplasts\n- Where photosynthesis happens.\n- Have grana.\n";
    let objectives = "## Learning objectives\n- Identify chloroplasts.\n- Explain the light reaction.\n- Apply the photosynthesis equation.\n";
    let worked = "## Worked example\n- A chloroplast captures 6 photons via the Light reaction, feeding ATP and NADPH into the Calvin cycle.\n";
    let misc = "## Common misconceptions\n- Plants eat soil.\n- The Calvin cycle needs sunlight directly.\n";
    for (path, content) in [
        ("class-notes-plan.md", plan),
        ("concept-1.md", concept_1),
        ("concept-2.md", concept_2),
        ("concept-3.md", concept_3),
        ("objectives.md", objectives),
        ("worked-example.md", worked),
        ("misconceptions.md", misc),
        ("homework.md", homework_md),
    ] {
        backend.push(
            MockScript::new()
                .text("Done.")
                .tool(
                    "Write",
                    serde_json::json!({"path":path,"content":content}),
                )
                .done(StopReason::Eos),
        );
    }

    let templates_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("templates/typst");
    let (flow, ctx) = flow_with_ctx_from_source(
        root,
        NaiveDate::from_ymd_opt(2026, 5, 15).unwrap(),
        ClassPlanSource::Text("Photosynthesis chapter".into()),
        ocr,
        pdf,
        templates_dir,
    )
    .expect("flow built");

    let tools = ToolRegistry::new()
        .register(Arc::new(gt_tools::ReadTool))
        .register(Arc::new(gt_tools::WriteTool))
        .register(Arc::new(gt_tools::EditTool));
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
    res.map(|_| ())
}

#[tokio::test]
async fn validator_accepts_well_formed_homework() {
    let hw = "# Homework — Photosynthesis\n## Practice problems\n1. Explain photolysis. (maps to: Light reaction)\n2. Compare C3 vs CAM. (maps to: Light reaction)\n3. Predict the effect of red light. (maps to: Light reaction)\n4. Identify reactants. (maps to: Light reaction)\n5. Diagram chloroplast. (maps to: Light reaction)\n## Reflection prompt\nWhy?\n## Suggested time\n30 minutes\n";
    let res = run_with_homework(hw).await;
    assert!(res.is_ok(), "well-formed homework should pass; got {res:?}");
}

#[tokio::test]
async fn validator_rejects_missing_suffix() {
    let hw = "# Homework — Photosynthesis\n## Practice problems\n1. Explain photolysis. (maps to: Light reaction)\n2. Compare C3 vs CAM.\n3. Predict the effect of red light. (maps to: Light reaction)\n## Reflection prompt\nWhy?\n";
    let res = run_with_homework(hw).await;
    let err = res.expect_err("missing suffix should fail");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("missing the `(maps to:") && msg.contains("Compare C3"),
        "error should call out the missing suffix and quote the bad line; got: {msg}"
    );
}

#[tokio::test]
async fn validator_ignores_non_numbered_lines() {
    // Bullet lines, headings, and prose lines are NOT validated — only lines
    // that start with `<digits>.` or `<digits>)`. Otherwise the validator
    // would mis-fire on the `## Reflection prompt` body or random prose.
    let hw = "# Homework\n## Practice problems\n1. one (maps to: Light reaction)\n2. two (maps to: Light reaction)\n3. three (maps to: Light reaction)\n4. four (maps to: Light reaction)\n5. five (maps to: Light reaction)\n## Reflection prompt\nWhat surprised you about this lesson?\n## Suggested time\n30 minutes\n- a stray bullet that is not a numbered problem\n";
    let res = run_with_homework(hw).await;
    assert!(res.is_ok(), "non-numbered lines must not trip the validator; got {res:?}");
}

#[tokio::test]
async fn validator_handles_two_digit_problems() {
    let hw = "# Homework\n## Practice problems\n1. one (maps to: Light reaction)\n2. two (maps to: Light reaction)\n3. three (maps to: Light reaction)\n4. four (maps to: Light reaction)\n5. five (maps to: Light reaction)\n10. ten (maps to: Light reaction)\n## Reflection prompt\nWhy?\n";
    let res = run_with_homework(hw).await;
    assert!(res.is_ok(), "two-digit numbering should validate; got {res:?}");

    let bad = "# Homework\n## Practice problems\n1. one (maps to: Light reaction)\n10. ten — missing suffix\n## Reflection prompt\nWhy?\n";
    let res = run_with_homework(bad).await;
    assert!(res.is_err(), "two-digit number with no suffix must fail");
}

#[tokio::test]
async fn validator_rejects_unknown_concept() {
    // class-notes has `### Light reaction`. The homework cites `Photolysis`,
    // which is not in class-notes — the validator must reject it. This locks
    // the live failure mode where the model invented a new concept name in
    // the tailored homework, causing the topic to drift.
    let hw = "# Homework\n## Practice problems\n1. one (maps to: Photolysis)\n2. two (maps to: Light reaction)\n3. three (maps to: Light reaction)\n4. four (maps to: Light reaction)\n5. five (maps to: Light reaction)\n## Reflection prompt\nWhy?\n";
    let res = run_with_homework(hw).await;
    let err = res.expect_err("unknown concept should fail");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("not in class-notes") && msg.contains("Photolysis"),
        "error should name the unknown concept and reference class-notes; got: {msg}"
    );
}
