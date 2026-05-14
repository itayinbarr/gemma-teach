//! `/student-edit <name>` — apply teacher's update notes to a student profile
//! and refresh their tags.
//!
//! Steps:
//!   1. delete-old-student-md  (det)
//!   2. rewrite-student        (agent: one-shot Write replacement)
//!   3. delete-old-tags        (det)
//!   4. refresh-tags           (agent: one-shot Write tags.json)
//!   5. validate-tags          (det)
//!
//! Single-step pattern throughout: each agent session receives the full prior
//! content in its task prompt and emits exactly one Write call. Multi-step
//! "Read then Edit/Write" dances are not reliable on Gemma 3n E2B.

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
            StepNode::det("delete-old-student-md", DeleteOldStudentMd),
            StepNode::agent("rewrite-student", RewriteStudent),
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

// ----- Step 1: snapshot + delete old student.md ------------------------------

struct DeleteOldStudentMd;
#[async_trait]
impl DeterministicStep for DeleteOldStudentMd {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let dir = student_dir(ctx)?;
        let path = dir.join(STUDENT_MD_FILENAME);
        if !path.exists() {
            return Err(FlowError::Step {
                step: "delete-old-student-md".into(),
                msg: format!(
                    "no student.md at {} — has this student been added yet?",
                    path.display()
                ),
            });
        }
        // Stash a backup so the agent prompt can include the prior content
        // even after we delete the file.
        let prior = tokio::fs::read_to_string(&path).await.map_err(|e| FlowError::Step {
            step: "delete-old-student-md".into(),
            msg: format!("read {}: {e}", path.display()),
        })?;
        let backup = dir.join(".student.md.prior");
        tokio::fs::write(&backup, prior.as_bytes())
            .await
            .map_err(|e| FlowError::Step {
                step: "delete-old-student-md".into(),
                msg: format!("write {}: {e}", backup.display()),
            })?;
        tokio::fs::remove_file(&path).await.map_err(|e| FlowError::Step {
            step: "delete-old-student-md".into(),
            msg: format!("rm {}: {e}", path.display()),
        })?;
        Ok(StepOutcome::default())
    }
}

// ----- Step 2: rewrite-student (agent, one-shot Write) -----------------------

const REWRITE_SYSTEM: &str = r##"You are a careful teaching assistant maintaining a student profile.

You can ONLY use this tool:
  - Write — creates a NEW file inside the working directory.

How to use tools:
  - Use `tool_code` fences to call tools, e.g.:
    ```tool_code
    Write(path="student.md", content="# Maya\n...")
    ```
  - One Write call is enough for this task. After Write succeeds, reply exactly: Done.

NON-NEGOTIABLE rules:
  - Preserve every detail from the prior profile that the teacher's notes do not change.
  - Apply the teacher's update notes precisely — do not invent additional changes.
  - Keep the same section structure as the prior profile.
"##;

struct RewriteStudent;
impl AgentStepFactory for RewriteStudent {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = student_dir(ctx).expect("student_dir");
        let prior = std::fs::read_to_string(dir.join(".student.md.prior"))
            .unwrap_or_else(|_| "(prior profile not found)".into());
        let edit_notes = ctx
            .inputs
            .get("edit_notes")
            .cloned()
            .unwrap_or_default();
        let task = format!(
            r#"Update this student's profile by writing a NEW `{STUDENT_MD_FILENAME}` that incorporates the teacher's notes below. Preserve everything from the prior profile that the notes do not change.

After Write succeeds, reply: Done.

--- prior student.md ---
{prior}
--- end of prior student.md ---

--- teacher's update notes ---
{edit_notes}
--- end of update notes ---"#
        );
        SessionBuilder::new("rewrite-student", dir)
            .system_prompt(REWRITE_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_3n_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![("student_md".into(), PathBuf::from(STUDENT_MD_FILENAME))]
    }
}

// ----- Step 3: delete old tags ------------------------------------------------

struct DeleteOldTags;
#[async_trait]
impl DeterministicStep for DeleteOldTags {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let dir = student_dir(ctx)?;
        let path = dir.join(TAGS_JSON_FILENAME);
        if path.exists() {
            tokio::fs::remove_file(&path)
                .await
                .map_err(|e| FlowError::Step {
                    step: "delete-old-tags".into(),
                    msg: format!("rm {}: {e}", path.display()),
                })?;
        }
        // Clean up the prior backup now that we've moved past rewrite-student.
        let backup = dir.join(".student.md.prior");
        if backup.exists() {
            let _ = tokio::fs::remove_file(&backup).await;
        }
        Ok(StepOutcome::default())
    }
}

// ----- Step 4: refresh-tags (agent, one-shot Write) ---------------------------

const REFRESH_SYSTEM: &str = r##"You are a careful teaching assistant.

You can ONLY use this tool:
  - Write — creates a NEW file inside the working directory.

How to use tools:
  - Use `tool_code` fences to call tools, e.g.:
    ```tool_code
    Write(path="tags.json", content="[\"anime\", \"marine-biology\"]")
    ```
  - One Write call is enough for this task. After Write succeeds, reply exactly: Done.

Output format for `tags.json` (MANDATORY):
A single JSON array, each element a string of one to three words separated by hyphens.
Examples of valid tags: "anime", "k-pop", "marine-biology", "competitive-chess".
Do NOT include explanations, code fences, or any other text in `tags.json`.
"##;

struct RefreshTags;
impl AgentStepFactory for RefreshTags {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = student_dir(ctx).expect("student_dir");
        let profile = std::fs::read_to_string(dir.join(STUDENT_MD_FILENAME))
            .unwrap_or_else(|_| "(student.md not found)".into());
        let task = format!(
            r#"Below is a student's profile. Extract 4-10 interest tags as lowercase kebab-case strings (one to three words each, e.g. "marine-biology", "studio-ghibli"), and write them as a JSON array to `{TAGS_JSON_FILENAME}`.

After Write succeeds, reply: Done.

--- student.md ---
{profile}
--- end of student.md ---"#
        );
        SessionBuilder::new("refresh-tags", dir)
            .system_prompt(REFRESH_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_3n_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![("tags_json".into(), PathBuf::from(TAGS_JSON_FILENAME))]
    }
}

// ----- Step 5: validate-tags --------------------------------------------------

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
