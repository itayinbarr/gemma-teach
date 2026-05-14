//! `/student-edit <name>` — update a student profile and refresh their tags.
//!
//! Steps:
//!   1. update-student   (agent: Read + Edit only — NOT Write; the file exists)
//!   2. delete-old-tags  (deterministic: remove tags.json so refresh-tags can Write)
//!   3. refresh-tags     (agent: Read + Write)
//!   4. validate-tags    (deterministic)

use async_trait::async_trait;
use chrono::NaiveDate;
use gt_core::session::SessionBuilder;
use std::path::PathBuf;

use crate::context::FlowCtx;
use crate::step::{AgentStepFactory, DeterministicStep, Flow, FlowError, StepNode, StepOutcome};
use crate::student_add::slugify;

const STUDENT_MD_FILENAME: &str = "student.md";
const TAGS_JSON_FILENAME: &str = "tags.json";

pub fn build_flow() -> Flow {
    Flow::new(
        "/student-edit".to_string(),
        vec![
            StepNode::agent("update-student", UpdateStudent),
            StepNode::det("delete-old-tags", DeleteOldTags),
            StepNode::agent("refresh-tags", RefreshTags),
            StepNode::det("validate-tags", ValidateTags),
        ],
    )
}

pub fn flow_with_ctx(
    root: PathBuf,
    date: NaiveDate,
    name: String,
    edit_notes: String,
) -> (Flow, FlowCtx) {
    let ctx = FlowCtx::new(&root, date)
        .with_input("name", name)
        .with_input("edit_notes", edit_notes);
    (build_flow(), ctx)
}

fn student_dir(ctx: &FlowCtx) -> Result<PathBuf, FlowError> {
    let name = ctx
        .inputs
        .get("name")
        .ok_or_else(|| FlowError::Internal("flow input 'name' missing".into()))?;
    Ok(ctx.root.join("students").join(slugify(name)))
}

// ----- Step 1: update-student -----------------------------------------------

const UPDATE_SYSTEM: &str = r#"You are a careful teaching assistant maintaining a student profile.

You can ONLY use these tools:
  - Read — reads `student.md` in the working directory.
  - Edit — replaces an EXACT block of text in `student.md`.

How to use tools:
  - Emit tool calls natively. Do NOT wrap in code fences or XML tags.
  - First Read `student.md`.
  - Then issue one or more Edit calls to apply the teacher's update notes.
  - Prefer many small Edits over one big rewrite — change only what the notes ask for.
  - After all edits succeed, reply: Done.

Notes on Edit:
  - `old_text` must match the file content EXACTLY (whitespace included).
  - Include enough surrounding context in `old_text` to make the match unique.
"#;

struct UpdateStudent;
impl AgentStepFactory for UpdateStudent {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = student_dir(ctx).expect("student_dir");
        let edit_notes = ctx
            .inputs
            .get("edit_notes")
            .cloned()
            .unwrap_or_default();
        let task = format!(
            r#"Read `student.md` and apply the teacher's update notes below using Edit.

Teacher's update notes:
---
{edit_notes}
---

After all edits succeed, reply: Done."#
        );
        SessionBuilder::new("update-student", dir)
            .system_prompt(UPDATE_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Read", "Edit"])
            .model_profile(gt_core::ModelProfile::gemma_3n_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![("student_md".into(), PathBuf::from(STUDENT_MD_FILENAME))]
    }
}

// ----- Step 2: delete-old-tags ----------------------------------------------

struct DeleteOldTags;
#[async_trait]
impl DeterministicStep for DeleteOldTags {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let dir = student_dir(ctx)?;
        let path = dir.join(TAGS_JSON_FILENAME);
        // Tolerate already-missing.
        if path.exists() {
            tokio::fs::remove_file(&path)
                .await
                .map_err(|e| FlowError::Step {
                    step: "delete-old-tags".into(),
                    msg: format!("rm {}: {e}", path.display()),
                })?;
        }
        Ok(StepOutcome::default())
    }
}

// ----- Step 3: refresh-tags --------------------------------------------------

const REFRESH_SYSTEM: &str = r#"You are a careful teaching assistant.

You can ONLY use these tools:
  - Read  — reads `student.md`.
  - Write — creates a NEW file `tags.json`.

How to use tools:
  - Emit tool calls natively. Do NOT wrap in code fences or XML tags.
  - First Read `student.md` to see the updated profile.
  - Then Write `tags.json` with a valid JSON array of lowercase kebab-case strings.
  - After Write succeeds, reply: Done.
"#;

struct RefreshTags;
impl AgentStepFactory for RefreshTags {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = student_dir(ctx).expect("student_dir");
        SessionBuilder::new("refresh-tags", dir)
            .system_prompt(REFRESH_SYSTEM)
            .task_prompt(
                "Read `student.md` and regenerate `tags.json`. Use lowercase kebab-case tags, 4-10 of them. After Write succeeds, reply: Done.".to_string(),
            )
            .allowed_tools(["Read", "Write"])
            .model_profile(gt_core::ModelProfile::gemma_3n_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![("tags_json".into(), PathBuf::from(TAGS_JSON_FILENAME))]
    }
}

// ----- Step 4: validate-tags -------------------------------------------------

struct ValidateTags;
#[async_trait]
impl DeterministicStep for ValidateTags {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let dir = student_dir(ctx)?;
        let path = dir.join(TAGS_JSON_FILENAME);
        let text = tokio::fs::read_to_string(&path).await.map_err(|e| FlowError::Step {
            step: "validate-tags".into(),
            msg: format!("read {}: {e}", path.display()),
        })?;
        let parsed: Vec<String> =
            serde_json::from_str(&text).map_err(|e| FlowError::Step {
                step: "validate-tags".into(),
                msg: format!("{TAGS_JSON_FILENAME} is not a JSON array of strings: {e}"),
            })?;
        if parsed.is_empty() {
            return Err(FlowError::Step {
                step: "validate-tags".into(),
                msg: format!("{TAGS_JSON_FILENAME} is empty"),
            });
        }
        Ok(StepOutcome::default())
    }
}
