//! `/student-add` — add a student to the class notebook.
//!
//! Steps:
//!   1. mk-student-dir   (deterministic)
//!   2. write-student    (agent: writes `student.md` from teacher notes)
//!   3. extract-tags     (agent: writes `tags.json` from the student.md)
//!   4. validate-tags    (deterministic: ensure tags.json parses as Vec<String>)
//!   5. tag-intersections (deterministic: compute overlaps with other students)
//!
//! Inputs (in `FlowCtx::inputs`): `name`, `description` (free-text raw notes).
//! Date is taken from `FlowCtx::date`.

use async_trait::async_trait;
use gt_core::session::SessionBuilder;
use std::path::PathBuf;
use std::sync::Arc;

use crate::context::FlowCtx;
use crate::step::{
    AgentStepFactory, DeterministicStep, Flow, FlowError, StepNode, StepOutcome,
};

const ARTIFACT_DIR: &str = "student_dir";
const ARTIFACT_STUDENT_MD: &str = "student_md";
const ARTIFACT_TAGS_JSON: &str = "tags_json";

/// Public entry point — build the flow with the teacher's inputs already set
/// on `FlowCtx`.
pub fn build_flow() -> Flow {
    Flow::new(
        "/student-add".to_string(),
        vec![
            StepNode::det("mk-student-dir", MkStudentDir),
            StepNode::agent("write-student", WriteStudent),
            StepNode::agent("extract-tags", ExtractTags),
            StepNode::det("validate-tags", ValidateTags),
            StepNode::det("tag-intersections", TagIntersections),
        ],
    )
}

// ----- Helpers ---------------------------------------------------------------

pub fn slugify(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut last_dash = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash && !out.is_empty() {
            out.push('-');
            last_dash = true;
        }
    }
    if out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        "unnamed".to_string()
    } else {
        out
    }
}

fn student_dir(ctx: &FlowCtx) -> Result<PathBuf, FlowError> {
    let name = ctx
        .inputs
        .get("name")
        .ok_or_else(|| FlowError::Internal("flow input 'name' missing".into()))?;
    Ok(ctx.root.join("students").join(slugify(name)))
}

// ----- Step 1: mk-student-dir -----------------------------------------------

struct MkStudentDir;
#[async_trait]
impl DeterministicStep for MkStudentDir {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let dir = student_dir(ctx)?;
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| FlowError::Step {
                step: "mk-student-dir".into(),
                msg: format!("mkdir {}: {e}", dir.display()),
            })?;
        Ok(StepOutcome {
            outputs: vec![(ARTIFACT_DIR.into(), dir)],
        })
    }
}

// ----- Step 2: write-student -------------------------------------------------

const STUDENT_MD_FILENAME: &str = "student.md";

const WRITE_STUDENT_SYSTEM: &str = r#"You are a careful teaching assistant working inside a fixed working directory.

You can ONLY use these tools:
  - Write — creates a NEW file inside the working directory.

How to use tools:
  - Emit tool calls natively. Do NOT wrap them in code fences or XML tags.
  - One tool call is enough for this task. After Write succeeds, reply exactly:
    Done.

Completion:
  - When the task is fully done, respond with the single word `Done.` and emit no more tool calls.
"#;

struct WriteStudent;

impl AgentStepFactory for WriteStudent {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = student_dir(ctx).expect("student_dir resolved");
        let name = ctx
            .inputs
            .get("name")
            .cloned()
            .unwrap_or_else(|| "Student".into());
        let date = ctx.date.format("%Y-%m-%d").to_string();
        let description = ctx
            .inputs
            .get("description")
            .cloned()
            .unwrap_or_default();

        let task = format!(
            r#"Create a profile file for the student below.

Required structure (in this order):

# {name}

## Snapshot
- 2-4 bullets summarizing this student.

## Interests
- 2-6 bullets of subjects, topics, or activities they care about.

## Hobbies
- 2-6 bullets of how they spend their time outside class.

## Media they love
- 2-6 bullets: shows, books, films, music, games.

## Notes for tailoring lessons
- 2-4 short bullets a teacher can use to make material feel personal to this student.

Be concrete and faithful to the notes below. Do not invent details that are not supported.

Date: {date}
Teacher's raw notes:
---
{description}
---

Write the file to `student.md` using the Write tool. After Write succeeds, reply: Done."#,
            name = name,
            date = date,
            description = description,
        );

        SessionBuilder::new("write-student", dir)
            .system_prompt(WRITE_STUDENT_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_3n_e2b())
    }

    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![(ARTIFACT_STUDENT_MD.into(), PathBuf::from(STUDENT_MD_FILENAME))]
    }
}

// ----- Step 3: extract-tags --------------------------------------------------

const TAGS_JSON_FILENAME: &str = "tags.json";

// Small local models tend to fail the multi-turn "Read → Write" dance: after
// the Read tool result, the model often emits EOS immediately. To work with
// Gemma's strength (single-turn generation) we pre-load `student.md` into the
// task prompt deterministically so the model only needs one Write call.
const EXTRACT_TAGS_SYSTEM: &str = r##"You are a careful teaching assistant working inside a fixed working directory.

You can ONLY use this tool:
  - Write — creates a NEW file inside the working directory.

How to use tools:
  - Use `tool_code` fences to call tools, e.g.:
    ```tool_code
    Write(path="tags.json", content="[\"anime\", \"marine-biology\"]")
    ```
  - One tool call is enough for this task. After Write succeeds, reply exactly: Done.

Output format for `tags.json` (MANDATORY):
A single JSON array, each element a string of one to three words separated by hyphens.
Examples of valid tags: "anime", "k-pop", "marine-biology", "competitive-chess".
Do NOT include explanations, code fences, or any other text in `tags.json`.
"##;

struct ExtractTags;

impl AgentStepFactory for ExtractTags {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = student_dir(ctx).expect("student_dir resolved");
        // Pre-load the student profile so the model has the data in-context.
        let profile = std::fs::read_to_string(dir.join(STUDENT_MD_FILENAME))
            .unwrap_or_else(|_| "(student.md not found)".into());
        let task = format!(
            r#"Below is a student's profile. Read it carefully, extract 4-10 interest tags as lowercase kebab-case strings (one to three words each, e.g. "marine-biology", "studio-ghibli"), and write them as a JSON array to `{TAGS_JSON_FILENAME}`.

After Write succeeds, reply: Done.

--- student.md ---
{profile}
--- end of student.md ---"#
        );
        SessionBuilder::new("extract-tags", dir)
            .system_prompt(EXTRACT_TAGS_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_3n_e2b())
    }

    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![(ARTIFACT_TAGS_JSON.into(), PathBuf::from(TAGS_JSON_FILENAME))]
    }
}

// ----- Step 4: validate-tags (deterministic) ---------------------------------

struct ValidateTags;
#[async_trait]
impl DeterministicStep for ValidateTags {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let dir = student_dir(ctx)?;
        let path = dir.join(TAGS_JSON_FILENAME);
        let text = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| FlowError::Step {
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
                msg: format!("{TAGS_JSON_FILENAME} is empty — re-run /student-add"),
            });
        }
        Ok(StepOutcome::default())
    }
}

// ----- Step 5: tag-intersections (deterministic) -----------------------------

struct TagIntersections;
#[async_trait]
impl DeterministicStep for TagIntersections {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let students_root = ctx.root.join("students");
        let dir = student_dir(ctx)?;
        let my_slug = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unnamed")
            .to_string();
        let my_tags = read_tags(&dir.join(TAGS_JSON_FILENAME)).await.unwrap_or_default();
        let my_set: std::collections::BTreeSet<String> = my_tags.into_iter().collect();

        let mut overlaps: Vec<(String, Vec<String>)> = Vec::new();
        let mut rd = match tokio::fs::read_dir(&students_root).await {
            Ok(r) => r,
            Err(_) => return Ok(StepOutcome::default()),
        };
        while let Ok(Some(entry)) = rd.next_entry().await {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let other_slug = match path.file_name().and_then(|s| s.to_str()) {
                Some(s) if s != my_slug => s.to_string(),
                _ => continue,
            };
            let their_tags = read_tags(&path.join(TAGS_JSON_FILENAME)).await.unwrap_or_default();
            let shared: Vec<String> =
                their_tags.into_iter().filter(|t| my_set.contains(t)).collect();
            if !shared.is_empty() {
                overlaps.push((other_slug, shared));
            }
        }
        overlaps.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
        let json = serde_json::to_string_pretty(&overlaps).unwrap_or_else(|_| "[]".into());
        let out_path = dir.join("intersections.json");
        tokio::fs::write(&out_path, json.as_bytes())
            .await
            .map_err(|e| FlowError::Step {
                step: "tag-intersections".into(),
                msg: format!("write {}: {e}", out_path.display()),
            })?;
        Ok(StepOutcome {
            outputs: vec![("intersections_json".into(), out_path)],
        })
    }
}

async fn read_tags(path: &std::path::Path) -> Option<Vec<String>> {
    let text = tokio::fs::read_to_string(path).await.ok()?;
    serde_json::from_str(&text).ok()
}

/// Build the flow plus apply the teacher's inputs to a fresh `FlowCtx`.
/// Convenience used by `gt-tui`.
pub fn flow_with_ctx(
    root: PathBuf,
    date: chrono::NaiveDate,
    name: String,
    description: String,
) -> (Flow, FlowCtx) {
    let ctx = FlowCtx::new(&root, date)
        .with_input("name", name)
        .with_input("description", description);
    (build_flow(), ctx)
}

#[allow(dead_code)]
fn _force_arc<T: Send + Sync>(_: Arc<T>) {}
