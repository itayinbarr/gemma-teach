//! `/class-plan <pdf>` — OCR a chapter, generate class notes + homework,
//! then re-skin both for every student through their interests, then compile
//! PDFs for the master and every per-student variant.

use async_trait::async_trait;
use chrono::NaiveDate;
use gt_core::session::SessionBuilder;
use gt_tools::{OcrRunner, PdfRunner};
use std::path::PathBuf;
use std::sync::Arc;

use crate::context::FlowCtx;
use crate::step::{AgentStepFactory, DeterministicStep, Flow, FlowError, StepNode, StepOutcome};

const LESSON_DIR_KEY: &str = "lesson_dir";
const SOURCE_TXT_KEY: &str = "source_txt";
const CLASS_NOTES_KEY: &str = "class_notes_md";
const HOMEWORK_KEY: &str = "homework_md";

const SOURCE_TXT_FILENAME: &str = "source.txt";
const CLASS_NOTES_FILENAME: &str = "class-notes.md";
const HOMEWORK_FILENAME: &str = "homework.md";
const STUDENT_NOTES_FILENAME: &str = "notes.md";
const STUDENT_HW_FILENAME: &str = "homework.md";

pub fn build_flow(
    ocr: Arc<dyn OcrRunner>,
    pdf: Arc<dyn PdfRunner>,
    templates_dir: PathBuf,
    student_slugs: Vec<String>,
) -> Flow {
    let mut steps = vec![
        StepNode::det("mk-lesson-dir", MkLessonDir),
        StepNode::det("ocr-source", OcrSource { ocr: ocr.clone() }),
        StepNode::agent("write-class-notes", WriteClassNotes),
        StepNode::agent("write-homework", WriteHomework),
    ];
    // Per-student steps. We copy context deterministically into the per-student dir,
    // then run the tailor session against that dir. Tailor sessions run under the
    // parallel group "tailor" so the orchestrator's semaphore caps concurrency.
    for slug in &student_slugs {
        steps.push(StepNode::det(
            format!("copy-context-for-{slug}"),
            CopyContextForStudent { slug: slug.clone() },
        ));
        steps.push(
            StepNode::agent(
                format!("tailor-for-{slug}"),
                TailorForStudent { slug: slug.clone() },
            )
            .in_group("tailor"),
        );
    }
    steps.push(StepNode::det(
        "compile-pdfs",
        CompilePdfs {
            pdf,
            templates_dir,
            student_slugs,
        },
    ));
    Flow::new("/class-plan".to_string(), steps)
}

/// Public convenience: build a flow plus a `FlowCtx` carrying the inputs.
pub fn flow_with_ctx(
    root: PathBuf,
    date: NaiveDate,
    pdf_path: PathBuf,
    ocr: Arc<dyn OcrRunner>,
    pdf: Arc<dyn PdfRunner>,
    templates_dir: PathBuf,
) -> Result<(Flow, FlowCtx), FlowError> {
    let students_dir = root.join("students");
    let mut slugs = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&students_dir) {
        for entry in rd.flatten() {
            if entry.path().is_dir() {
                if let Some(s) = entry.file_name().to_str() {
                    slugs.push(s.to_string());
                }
            }
        }
    }
    slugs.sort();

    let ctx = FlowCtx::new(&root, date).with_input("pdf_path", pdf_path.display().to_string());
    let flow = build_flow(ocr, pdf, templates_dir, slugs);
    Ok((flow, ctx))
}

// ----- Step 1: mk-lesson-dir ------------------------------------------------

fn lesson_dir(ctx: &FlowCtx) -> PathBuf {
    ctx.root
        .join("lessons")
        .join(ctx.date.format("%Y-%m-%d").to_string())
}

struct MkLessonDir;
#[async_trait]
impl DeterministicStep for MkLessonDir {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let dir = lesson_dir(ctx);
        tokio::fs::create_dir_all(dir.join("per-student"))
            .await
            .map_err(|e| FlowError::Step {
                step: "mk-lesson-dir".into(),
                msg: e.to_string(),
            })?;
        Ok(StepOutcome {
            outputs: vec![(LESSON_DIR_KEY.into(), dir)],
        })
    }
}

// ----- Step 2: ocr-source ---------------------------------------------------

struct OcrSource {
    ocr: Arc<dyn OcrRunner>,
}
#[async_trait]
impl DeterministicStep for OcrSource {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let pdf = ctx
            .inputs
            .get("pdf_path")
            .ok_or_else(|| FlowError::Internal("flow input 'pdf_path' missing".into()))?;
        let dir = lesson_dir(ctx);
        let out = dir.join(SOURCE_TXT_FILENAME);
        self.ocr
            .ocr_pdf_to_text(std::path::Path::new(pdf), &out)
            .await
            .map_err(|e| FlowError::Step {
                step: "ocr-source".into(),
                msg: e.to_string(),
            })?;
        Ok(StepOutcome {
            outputs: vec![(SOURCE_TXT_KEY.into(), out)],
        })
    }
}

// ----- Step 3: write-class-notes -------------------------------------------

const WRITE_NOTES_SYSTEM: &str = r#"You are a careful teaching assistant working inside a fixed working directory.

You can ONLY use these tools:
  - Read  — reads an existing file in the working directory.
  - Write — creates a NEW file in the working directory.

How to use tools:
  - Emit tool calls natively. Do NOT wrap in code fences or XML tags.
  - First Read `source.txt` to see the OCR'd chapter.
  - Then Write `class-notes.md` with the required structure.
  - After Write succeeds, reply: Done.
"#;

struct WriteClassNotes;
impl AgentStepFactory for WriteClassNotes {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = lesson_dir(ctx);
        let task = format!(
            r#"Read `source.txt`. It is the OCR'd content of a textbook chapter.

Produce `class-notes.md` with EXACTLY this structure:

# <title — infer from source>

## Learning objectives
- 3-5 bullets, each starting with a verb (identify, explain, apply, contrast, predict).

## Key concepts
### <concept 1>
- 2-4 bullets explaining it concretely.

### <concept 2>
- 2-4 bullets.

### <concept 3>
- 2-4 bullets.

## Worked example
- A single concrete example that uses these concepts.

## Common misconceptions
- 2-4 bullets a teacher should pre-empt.

Stay strictly faithful to `source.txt`. Do not introduce material that is not in the source.
After Write succeeds, reply: Done."#
        );
        SessionBuilder::new("write-class-notes", dir)
            .system_prompt(WRITE_NOTES_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Read", "Write"])
            .model_profile(gt_core::ModelProfile::gemma_3n_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![(CLASS_NOTES_KEY.into(), PathBuf::from(CLASS_NOTES_FILENAME))]
    }
}

// ----- Step 4: write-homework ----------------------------------------------

const WRITE_HW_SYSTEM: &str = WRITE_NOTES_SYSTEM;

struct WriteHomework;
impl AgentStepFactory for WriteHomework {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = lesson_dir(ctx);
        let task = r#"Read `class-notes.md`. Produce `homework.md` with EXACTLY this structure:

# Homework — <same title as class-notes.md>

## Practice problems
1. <problem mapped to a concept from ## Key concepts>
2. <problem>
3. <problem>
4. <problem>
5. <problem>

## Reflection prompt
One short open-ended question.

## Suggested time
e.g., "30 minutes"

Every problem MUST map to a concept from `## Key concepts` of `class-notes.md`. Problems should grow in difficulty.
After Write succeeds, reply: Done."#;
        SessionBuilder::new("write-homework", dir)
            .system_prompt(WRITE_HW_SYSTEM)
            .task_prompt(task.to_string())
            .allowed_tools(["Read", "Write"])
            .model_profile(gt_core::ModelProfile::gemma_3n_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![(HOMEWORK_KEY.into(), PathBuf::from(HOMEWORK_FILENAME))]
    }
}

// ----- Step 5: copy-context-for-<student> ----------------------------------

struct CopyContextForStudent {
    slug: String,
}
#[async_trait]
impl DeterministicStep for CopyContextForStudent {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let lesson = lesson_dir(ctx);
        let dest = lesson.join("per-student").join(&self.slug);
        tokio::fs::create_dir_all(&dest).await.map_err(|e| FlowError::Step {
            step: format!("copy-context-for-{}", self.slug),
            msg: e.to_string(),
        })?;
        // class-notes.md + homework.md
        for f in [CLASS_NOTES_FILENAME, HOMEWORK_FILENAME] {
            let from = lesson.join(f);
            let to = dest.join(f);
            tokio::fs::copy(&from, &to).await.map_err(|e| FlowError::Step {
                step: format!("copy-context-for-{}", self.slug),
                msg: format!("copy {} -> {}: {e}", from.display(), to.display()),
            })?;
        }
        // student.md + tags.json from the student's profile
        for f in ["student.md", "tags.json"] {
            let from = ctx.root.join("students").join(&self.slug).join(f);
            let to = dest.join(f);
            tokio::fs::copy(&from, &to).await.map_err(|e| FlowError::Step {
                step: format!("copy-context-for-{}", self.slug),
                msg: format!("copy {} -> {}: {e}", from.display(), to.display()),
            })?;
        }
        Ok(StepOutcome::default())
    }
}

// ----- Step 6: tailor-for-<student> (parallel group "tailor") ---------------

const TAILOR_SYSTEM: &str = r#"You are a careful teaching assistant tailoring a lesson for one student.

You can ONLY use these tools:
  - Read  — reads an existing file in the working directory.
  - Write — creates a NEW file in the working directory.

How to use tools:
  - Emit tool calls natively. Do NOT wrap in code fences or XML tags.
  - Read `student.md`, `tags.json`, `class-notes.md`, and `homework.md` in that order.
  - Then Write `notes.md` first, then Write `homework.md`.
  - After both Writes succeed, reply: Done.

NON-NEGOTIABLE rules:
  - Cover the SAME concepts and the SAME learning objectives as `class-notes.md`.
  - Keep the SAME problem count as `homework.md` (no more, no less).
  - Re-skin examples, analogies, and problem framings using the student's interests.
  - Keep the same section headings as the originals.
"#;

struct TailorForStudent {
    slug: String,
}
impl AgentStepFactory for TailorForStudent {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = lesson_dir(ctx).join("per-student").join(&self.slug);
        let task = r#"Tailor today's lesson for this student.

Read these files in order:
  1. student.md
  2. tags.json
  3. class-notes.md
  4. homework.md

Then produce TWO files in this directory:
  - notes.md     — same structure as class-notes.md, re-skinned for this student.
  - homework.md  — same problem count as homework.md, re-skinned for this student.

After both Writes succeed, reply: Done."#;
        SessionBuilder::new(format!("tailor-for-{}", self.slug), dir)
            .system_prompt(TAILOR_SYSTEM)
            .task_prompt(task.to_string())
            .allowed_tools(["Read", "Write"])
            .model_profile(gt_core::ModelProfile::gemma_3n_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![
            (
                format!("tailored_notes_{}", self.slug),
                PathBuf::from(STUDENT_NOTES_FILENAME),
            ),
            (
                format!("tailored_hw_{}", self.slug),
                PathBuf::from(STUDENT_HW_FILENAME),
            ),
        ]
    }
}

// ----- Step 7: compile-pdfs -------------------------------------------------

struct CompilePdfs {
    pdf: Arc<dyn PdfRunner>,
    templates_dir: PathBuf,
    student_slugs: Vec<String>,
}
#[async_trait]
impl DeterministicStep for CompilePdfs {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let lesson = lesson_dir(ctx);
        let notes_tpl = self.templates_dir.join("notes.typ");
        let hw_tpl = self.templates_dir.join("homework.typ");

        let mut outputs: Vec<(String, PathBuf)> = Vec::new();
        let compile_one = |md: PathBuf, tpl: PathBuf, key: String| -> _ {
            let pdf = self.pdf.clone();
            async move {
                let out = md.with_extension("pdf");
                match pdf.compile(&md, &tpl, &out).await {
                    Ok(()) => Ok::<_, FlowError>((key, out)),
                    Err(e) => Err(FlowError::Step {
                        step: "compile-pdfs".into(),
                        msg: format!("typst {}: {e}", md.display()),
                    }),
                }
            }
        };

        let (k, p) = compile_one(
            lesson.join(CLASS_NOTES_FILENAME),
            notes_tpl.clone(),
            "class_notes_pdf".into(),
        )
        .await?;
        outputs.push((k, p));
        let (k, p) = compile_one(
            lesson.join(HOMEWORK_FILENAME),
            hw_tpl.clone(),
            "homework_pdf".into(),
        )
        .await?;
        outputs.push((k, p));

        for slug in &self.student_slugs {
            let dir = lesson.join("per-student").join(slug);
            let (k, p) = compile_one(
                dir.join(STUDENT_NOTES_FILENAME),
                notes_tpl.clone(),
                format!("tailored_notes_pdf_{slug}"),
            )
            .await?;
            outputs.push((k, p));
            let (k, p) = compile_one(
                dir.join(STUDENT_HW_FILENAME),
                hw_tpl.clone(),
                format!("tailored_hw_pdf_{slug}"),
            )
            .await?;
            outputs.push((k, p));
        }
        Ok(StepOutcome { outputs })
    }
}
