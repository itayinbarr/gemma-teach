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

/// Where the class-plan source content comes from. Three accepted shapes:
/// a PDF file (OCR'd via Tesseract), a plain-text file (used as-is), or
/// inline text (typically pasted in the TUI).
#[derive(Debug, Clone)]
pub enum ClassPlanSource {
    Pdf(PathBuf),
    TextFile(PathBuf),
    Text(String),
}

impl ClassPlanSource {
    /// Detect the source kind from a path extension. `.pdf` is OCR'd; anything
    /// else is treated as a plain text file.
    pub fn from_path(p: PathBuf) -> Self {
        if p.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("pdf")) == Some(true) {
            Self::Pdf(p)
        } else {
            Self::TextFile(p)
        }
    }
}

const LESSON_DIR_KEY: &str = "lesson_dir";
const SOURCE_TXT_KEY: &str = "source_txt";
const CLASS_NOTES_KEY: &str = "class_notes_md";
const HOMEWORK_KEY: &str = "homework_md";

const SOURCE_TXT_FILENAME: &str = "source.txt";
const CLASS_NOTES_FILENAME: &str = "class-notes.md";
const HOMEWORK_FILENAME: &str = "homework.md";
const STUDENT_HW_FILENAME: &str = "homework.md";

// Class-notes part filenames. Each is written by a separate small agent
// session; assemble-class-notes deterministically concatenates them into the
// final `class-notes.md` the rest of the flow consumes. This decomposition
// replaces the single `write-class-notes` step that consistently failed on
// dense source material (the model would lose structure when asked to emit
// all six sections — title, objectives, three concept blocks, worked
// example, misconceptions — in one shot).
const CLASS_NOTES_PLAN_FILENAME: &str = "class-notes-plan.md";
const OBJECTIVES_FILENAME: &str = "objectives.md";
const WORKED_EXAMPLE_FILENAME: &str = "worked-example.md";
const MISCONCEPTIONS_FILENAME: &str = "misconceptions.md";
fn concept_filename(n: u32) -> String {
    format!("concept-{n}.md")
}
const CLASS_NOTES_CONCEPT_COUNT: u32 = 3;

pub fn build_flow(
    ocr: Arc<dyn OcrRunner>,
    pdf: Arc<dyn PdfRunner>,
    templates_dir: PathBuf,
    student_slugs: Vec<String>,
    source: ClassPlanSource,
) -> Flow {
    let prep_step: Box<dyn DeterministicStep> = match source {
        ClassPlanSource::Pdf(p) => Box::new(OcrSource { ocr: ocr.clone(), pdf_path: p }),
        ClassPlanSource::TextFile(p) => Box::new(LoadTextSource::FromFile(p)),
        ClassPlanSource::Text(t) => Box::new(LoadTextSource::Inline(t)),
    };
    let mut steps = vec![
        StepNode::det("mk-lesson-dir", MkLessonDir),
        StepNode {
            id: gt_core::ids::StepId::new(),
            name: "prepare-source".into(),
            kind: crate::step::StepKind::Deterministic(prep_step),
            parallel_group: None,
        },
        // ---- decomposed class-notes pipeline ----
        // plan → validate → 3 concept summaries + objectives + worked-ex +
        // misconceptions → deterministic assemble.
        StepNode::agent("plan-class-notes", PlanClassNotes),
        StepNode::det("validate-class-notes-plan", ValidateClassNotesPlan),
    ];
    for n in 1..=CLASS_NOTES_CONCEPT_COUNT {
        steps.push(
            StepNode::agent(format!("summarize-concept-{n}"), SummarizeConcept { n })
                .in_group("class-notes-parts"),
        );
    }
    steps.push(
        StepNode::agent("write-class-notes-objectives", WriteObjectives)
            .in_group("class-notes-parts"),
    );
    steps.push(
        StepNode::agent("write-class-notes-worked-example", WriteWorkedExample)
            .in_group("class-notes-parts"),
    );
    steps.push(
        StepNode::agent("write-class-notes-misconceptions", WriteMisconceptions)
            .in_group("class-notes-parts"),
    );
    steps.push(StepNode::det("assemble-class-notes", AssembleClassNotes));
    steps.push(StepNode::agent("write-homework", WriteHomework));
    steps.push(StepNode::det(
        "validate-homework-mapping",
        ValidateHomeworkMapping {
            path: HOMEWORK_FILENAME.into(),
            source: HomeworkSource::Master,
        },
    ));
    // Per-student steps. We copy context deterministically into the per-student dir,
    // then run the tailor session against that dir. Tailor sessions run under the
    // parallel group "tailor" so the orchestrator's semaphore caps concurrency.
    for slug in &student_slugs {
        // Per-student pipeline. Master class-notes are shared across all
        // students — mirrors real teaching practice where teachers share
        // notes and differentiate via assignments. Each student gets a
        // tailoring plan + a personalized homework sheet:
        //
        //   mk-dir-for-<slug>              (det)
        //   plan-tailoring-for-<slug>      (agent — emits tailoring-plan.md)
        //   validate-tailoring-plan-<slug> (det — checks plan shape)
        //   tailor-hw-for-<slug>           (agent — uses plan to write homework.md)
        //   validate-tailored-hw-<slug>    (det — concept-set + mapping)
        //   validate-tailor-divergence-<slug> (det — homework must differ from master)
        steps.push(StepNode::det(
            format!("mk-dir-for-{slug}"),
            MkPerStudentDir { slug: slug.clone() },
        ));
        steps.push(
            StepNode::agent(
                format!("plan-tailoring-for-{slug}"),
                PlanTailoring { slug: slug.clone() },
            )
            .in_group("tailor"),
        );
        steps.push(StepNode::det(
            format!("validate-tailoring-plan-{slug}"),
            ValidateTailoringPlan { slug: slug.clone() },
        ));
        steps.push(
            StepNode::agent(
                format!("tailor-hw-for-{slug}"),
                TailorHomeworkForStudent { slug: slug.clone() },
            )
            .in_group("tailor"),
        );
        steps.push(StepNode::det(
            format!("restore-hw-suffixes-{slug}"),
            RestoreHomeworkSuffixes { slug: slug.clone() },
        ));
        steps.push(StepNode::det(
            format!("validate-tailored-hw-{slug}"),
            ValidateHomeworkMapping {
                path: STUDENT_HW_FILENAME.into(),
                source: HomeworkSource::PerStudent { slug: slug.clone() },
            },
        ));
        steps.push(StepNode::det(
            format!("validate-tailor-divergence-{slug}"),
            ValidateTailorDivergence {
                slug: slug.clone(),
                min_change_ratio: 0.30,
            },
        ));
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

/// Public convenience: build a flow + a `FlowCtx` from any `ClassPlanSource`.
pub fn flow_with_ctx_from_source(
    root: PathBuf,
    date: NaiveDate,
    source: ClassPlanSource,
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
    let ctx = FlowCtx::new(&root, date);
    let flow = build_flow(ocr, pdf, templates_dir, slugs, source);
    Ok((flow, ctx))
}

/// Back-compat shim: build with a PDF path.
pub fn flow_with_ctx(
    root: PathBuf,
    date: NaiveDate,
    pdf_path: PathBuf,
    ocr: Arc<dyn OcrRunner>,
    pdf: Arc<dyn PdfRunner>,
    templates_dir: PathBuf,
) -> Result<(Flow, FlowCtx), FlowError> {
    flow_with_ctx_from_source(root, date, ClassPlanSource::Pdf(pdf_path), ocr, pdf, templates_dir)
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
    pdf_path: PathBuf,
}
#[async_trait]
impl DeterministicStep for OcrSource {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let dir = lesson_dir(ctx);
        let out = dir.join(SOURCE_TXT_FILENAME);
        self.ocr
            .ocr_pdf_to_text(&self.pdf_path, &out)
            .await
            .map_err(|e| FlowError::Step {
                step: "prepare-source".into(),
                msg: e.to_string(),
            })?;
        Ok(StepOutcome {
            outputs: vec![(SOURCE_TXT_KEY.into(), out)],
        })
    }
}

enum LoadTextSource {
    FromFile(PathBuf),
    Inline(String),
}
#[async_trait]
impl DeterministicStep for LoadTextSource {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let dir = lesson_dir(ctx);
        let out = dir.join(SOURCE_TXT_FILENAME);
        let text = match self {
            LoadTextSource::FromFile(p) => {
                tokio::fs::read_to_string(p).await.map_err(|e| FlowError::Step {
                    step: "prepare-source".into(),
                    msg: format!("read {}: {e}", p.display()),
                })?
            }
            LoadTextSource::Inline(t) => t.clone(),
        };
        tokio::fs::write(&out, text.as_bytes())
            .await
            .map_err(|e| FlowError::Step {
                step: "prepare-source".into(),
                msg: format!("write {}: {e}", out.display()),
            })?;
        Ok(StepOutcome {
            outputs: vec![(SOURCE_TXT_KEY.into(), out)],
        })
    }
}

// ----- Step 3: decomposed class-notes pipeline -----------------------------
//
// The single "write class-notes" task asked Gemma 4 to emit six structural
// sections in one shot (title, objectives, three concept blocks, worked
// example, misconceptions). On dense math source material the model
// consistently lost structure — collapsing sections, dropping concepts,
// echoing the prompt template, or emitting partial output that failed the
// downstream homework mapping check because the `### <concept>` headings
// were missing or mangled.
//
// The fix: scaffold-model fit. Decompose into:
//   1) plan-class-notes — emit a small plan (title + 3 concept names)
//   2) validate-class-notes-plan — parse the plan deterministically
//   3) summarize-concept-N × 3 — each writes ONE `### <name>\n- bullets` file
//   4) write-class-notes-objectives — one small file
//   5) write-class-notes-worked-example — one small file
//   6) write-class-notes-misconceptions — one small file
//   7) assemble-class-notes — deterministic concatenation into class-notes.md
// Each agent step now has a single bounded output the model can succeed at.

// One-shot pattern: each session gets its inputs pre-loaded into the task
// prompt and only needs to emit ONE Write tool call. This works with Gemma 4's
// single-turn strength instead of fighting it through multi-step Read+Write.
const ONE_SHOT_WRITE_SYSTEM: &str = r##"You are a careful teaching assistant working inside a fixed working directory.

You can ONLY use this tool:
  - Write — creates a NEW file inside the working directory.

How to use tools:
  - Use `tool_code` fences to call tools, e.g.:
    ```tool_code
    Write(path="<the path given in your task>", content="<the content described in your task>")
    ```
    Done.
  - The `path` argument MUST be the exact filename named in your task. Do not invent a different filename.
  - Output exactly ONE Write call inside a `tool_code` fence. After the closing ``` of the fence, emit a new line containing only: Done.
  - Do NOT emit any other tool call. Do NOT "verify" the file by reading it back. After "Done." output nothing more.
"##;

// ---- Step 3a: plan-class-notes (small, structured) ------------------------

struct PlanClassNotes;
impl AgentStepFactory for PlanClassNotes {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = lesson_dir(ctx);
        let source = std::fs::read_to_string(dir.join(SOURCE_TXT_FILENAME))
            .unwrap_or_else(|_| "(source.txt not found)".into());
        let task = format!(
            r#"You will write the file `{CLASS_NOTES_PLAN_FILENAME}` by calling the Write tool. Do NOT reply "Done." before calling Write — the file must exist first.

The file body has these sections, indented here for clarity (the file itself is not indented):

    # Class-notes plan

    ## Title
    <a short title for the chapter, inferred from the source>

    ## Concepts
    - concept: <name of concept 1>
    - concept: <name of concept 2>
    - concept: <name of concept 3>

Rules:
  - Exactly {CLASS_NOTES_CONCEPT_COUNT} concepts. Concept names are short, distinct, and named directly in the source (e.g. "Equivalent Fractions", "Ratios", "Chloroplasts").
  - Replace every <...> placeholder with real content drawn from the source.
  - No prose, no extra sections, no JSON.

The source you must read:

--- source.txt ---
{source}
--- end of source.txt ---

Now call Write with the filled-in content."#
        );
        SessionBuilder::new("plan-class-notes", dir)
            .system_prompt(ONE_SHOT_WRITE_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_4_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![(
            "class_notes_plan_md".into(),
            PathBuf::from(CLASS_NOTES_PLAN_FILENAME),
        )]
    }
}

// ---- Step 3b: validate-class-notes-plan (deterministic) -------------------

/// Plan parsed from the class-notes plan file. Two fields, both required.
#[derive(Debug, Default, Clone)]
pub struct ClassNotesPlan {
    pub title: String,
    pub concepts: Vec<String>,
}

/// Parse the markdown plan format:
///
/// ```text
/// ## Title
/// Fractions and Ratios
///
/// ## Concepts
/// - concept: Equivalent Fractions
/// - concept: Ratios
/// - concept: Fractions
/// ```
///
/// Tolerant of: missing `# Class-notes plan` heading, the model writing the
/// title on the same line as `## Title`, indent variations, list dashes.
pub fn parse_class_notes_plan(s: &str) -> Result<ClassNotesPlan, String> {
    enum Sec {
        None,
        Title,
        Concepts,
    }
    let mut sec = Sec::None;
    let mut plan = ClassNotesPlan::default();
    for raw in s.lines() {
        let line = raw.trim_end();
        let lower = line.trim().to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("## title") {
            sec = Sec::Title;
            // Allow inline title: "## Title: Fractions" or "## Title Fractions"
            let inline = rest.trim_start_matches([':', ' ']).trim();
            if !inline.is_empty() && plan.title.is_empty() {
                // Preserve original casing from the raw line.
                let raw_lower = line.to_ascii_lowercase();
                let idx = raw_lower.find("## title").unwrap() + "## title".len();
                plan.title = line[idx..].trim_start_matches([':', ' ']).trim().to_string();
            }
            continue;
        }
        if lower.starts_with("## concepts") {
            sec = Sec::Concepts;
            continue;
        }
        if lower.starts_with("## ") {
            sec = Sec::None;
            continue;
        }
        match sec {
            Sec::None => {}
            Sec::Title => {
                let t = line.trim();
                if !t.is_empty() && !t.starts_with('#') && plan.title.is_empty() {
                    plan.title = t.to_string();
                }
            }
            Sec::Concepts => {
                let t = line.trim().trim_start_matches('-').trim();
                if let Some(rest) = t.strip_prefix("concept:") {
                    let name = rest.trim();
                    if !name.is_empty() {
                        plan.concepts.push(name.to_string());
                    }
                } else if !t.is_empty()
                    && !t.starts_with('<')
                    && !t.starts_with("- ")
                    && line.trim_start().starts_with('-')
                {
                    // Fallback: model wrote `- Equivalent Fractions` without
                    // the `concept:` key.
                    plan.concepts.push(t.to_string());
                }
            }
        }
    }
    if plan.title.is_empty() {
        return Err(format!("{CLASS_NOTES_PLAN_FILENAME} is missing a title"));
    }
    if plan.concepts.len() < CLASS_NOTES_CONCEPT_COUNT as usize {
        return Err(format!(
            "{CLASS_NOTES_PLAN_FILENAME} has only {} concept(s), expected {CLASS_NOTES_CONCEPT_COUNT}",
            plan.concepts.len()
        ));
    }
    // Drop extras: downstream steps assume exactly N concepts.
    plan.concepts.truncate(CLASS_NOTES_CONCEPT_COUNT as usize);
    Ok(plan)
}

struct ValidateClassNotesPlan;
#[async_trait]
impl DeterministicStep for ValidateClassNotesPlan {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let path = lesson_dir(ctx).join(CLASS_NOTES_PLAN_FILENAME);
        let text = tokio::fs::read_to_string(&path).await.map_err(|e| FlowError::Step {
            step: "validate-class-notes-plan".into(),
            msg: format!("read {}: {e}", path.display()),
        })?;
        let plan = parse_class_notes_plan(&text).map_err(|m| FlowError::Step {
            step: "validate-class-notes-plan".into(),
            msg: m,
        })?;
        for (i, c) in plan.concepts.iter().enumerate() {
            if c.starts_with('<') || c.eq_ignore_ascii_case("concept") {
                return Err(FlowError::Step {
                    step: "validate-class-notes-plan".into(),
                    msg: format!(
                        "{CLASS_NOTES_PLAN_FILENAME} concept #{} echoes the placeholder: '{c}'",
                        i + 1
                    ),
                });
            }
        }
        Ok(StepOutcome::default())
    }
}

// ---- Helpers shared by per-part agents ------------------------------------

fn read_plan_or_default(ctx: &FlowCtx) -> ClassNotesPlan {
    let dir = lesson_dir(ctx);
    let text = std::fs::read_to_string(dir.join(CLASS_NOTES_PLAN_FILENAME)).unwrap_or_default();
    parse_class_notes_plan(&text).unwrap_or_default()
}

fn read_source(ctx: &FlowCtx) -> String {
    let dir = lesson_dir(ctx);
    std::fs::read_to_string(dir.join(SOURCE_TXT_FILENAME))
        .unwrap_or_else(|_| "(source.txt not found)".into())
}

// ---- Step 3c: summarize-concept-N -----------------------------------------

struct SummarizeConcept {
    n: u32,
}
impl AgentStepFactory for SummarizeConcept {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = lesson_dir(ctx);
        let plan = read_plan_or_default(ctx);
        let source = read_source(ctx);
        let concept = plan
            .concepts
            .get((self.n - 1) as usize)
            .cloned()
            .unwrap_or_else(|| format!("Concept {}", self.n));
        let n = self.n;
        let filename = concept_filename(n);
        let task = format!(
            r#"You will write the file `{filename}` by calling the Write tool. Do NOT reply "Done." before calling Write — the file must exist first.

The file is ONE concept only: "{concept}". Summarize the SOURCE below into a 2-4 bullet block about this one concept.

The file body has this shape, indented here for clarity (the file itself is not indented):

    ### {concept}
    - <bullet 1>
    - <bullet 2>
    - <bullet 3, optional>
    - <bullet 4, optional>

Rules:
  - 2-4 bullets. Each bullet a complete sentence, concrete, grounded in the source.
  - At least one bullet must include a NAMED entity or NUMERICAL value taken directly from the source (e.g. "3/4", "6:2", "75%"). Do not paraphrase the numbers.
  - Heading is exactly `### {concept}` — same casing as given, three hashes, no extra punctuation.
  - Do not write about anything other than "{concept}".

The source you must read:

--- source.txt ---
{source}
--- end of source.txt ---

Now call Write with the filled-in content."#,
        );
        SessionBuilder::new(format!("summarize-concept-{n}"), dir)
            .system_prompt(ONE_SHOT_WRITE_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_4_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![(
            format!("concept_{}_md", self.n),
            PathBuf::from(concept_filename(self.n)),
        )]
    }
}

// ---- Step 3d: write-class-notes-objectives --------------------------------

struct WriteObjectives;
impl AgentStepFactory for WriteObjectives {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = lesson_dir(ctx);
        let plan = read_plan_or_default(ctx);
        let source = read_source(ctx);
        let concepts = plan.concepts.join(", ");
        let task = format!(
            r#"You will write the file `{OBJECTIVES_FILENAME}` by calling the Write tool. Do NOT reply "Done." before calling Write — the file must exist first.

The file is just the Learning objectives section for a chapter whose concepts are: {concepts}.

The file body has this shape, indented here for clarity (the file itself is not indented):

    ## Learning objectives
    - <verb> <objective 1>
    - <verb> <objective 2>
    - <verb> <objective 3>
    - <verb> <objective 4, optional>
    - <verb> <objective 5, optional>

Rules:
  - 3-5 bullets. Each starts with a present-tense verb: identify, explain, apply, contrast, predict, compare, compute, simplify.
  - Each objective references one of the concepts above by name where natural.
  - One sentence per bullet, no sub-bullets.

The source you must read:

--- source.txt ---
{source}
--- end of source.txt ---

Now call Write with the filled-in content."#
        );
        SessionBuilder::new("write-class-notes-objectives", dir)
            .system_prompt(ONE_SHOT_WRITE_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_4_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![("objectives_md".into(), PathBuf::from(OBJECTIVES_FILENAME))]
    }
}

// ---- Step 3e: write-class-notes-worked-example ----------------------------

struct WriteWorkedExample;
impl AgentStepFactory for WriteWorkedExample {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = lesson_dir(ctx);
        let plan = read_plan_or_default(ctx);
        let source = read_source(ctx);
        let concepts = plan.concepts.join(", ");
        let task = format!(
            r#"You will write the file `{WORKED_EXAMPLE_FILENAME}` by calling the Write tool. Do NOT reply "Done." before calling Write — the file must exist first.

The file is just the Worked example section for a chapter whose concepts are: {concepts}.

The file body has this shape, indented here for clarity (the file itself is not indented):

    ## Worked example
    - <one concrete worked example, 1-3 sentences>

Rules:
  - Exactly one bullet under `## Worked example`. One bullet = one connected mini-explanation.
  - Use a NAMED entity or NUMERICAL value pulled directly from the source — do not paraphrase the numbers.
  - Reference at least two of the concepts above by name inside the bullet.

The source you must read:

--- source.txt ---
{source}
--- end of source.txt ---

Now call Write with the filled-in content."#
        );
        SessionBuilder::new("write-class-notes-worked-example", dir)
            .system_prompt(ONE_SHOT_WRITE_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_4_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![(
            "worked_example_md".into(),
            PathBuf::from(WORKED_EXAMPLE_FILENAME),
        )]
    }
}

// ---- Step 3f: write-class-notes-misconceptions ----------------------------

struct WriteMisconceptions;
impl AgentStepFactory for WriteMisconceptions {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = lesson_dir(ctx);
        let plan = read_plan_or_default(ctx);
        let source = read_source(ctx);
        let concepts = plan.concepts.join(", ");
        let task = format!(
            r#"You will write the file `{MISCONCEPTIONS_FILENAME}` by calling the Write tool. Do NOT reply "Done." before calling Write — the file must exist first.

The file is just the Common misconceptions section for a chapter whose concepts are: {concepts}.

The file body has this shape, indented here for clarity (the file itself is not indented):

    ## Common misconceptions
    - <misconception 1>
    - <misconception 2>
    - <misconception 3, optional>
    - <misconception 4, optional>

Rules:
  - 2-4 bullets. Each is a single sentence stating a wrong belief students commonly hold about one of the concepts above.
  - Phrase the bullet AS the wrong belief itself, not as a correction. The teacher will use these to pre-empt errors.
  - Each misconception must be grounded in the source (it should be a wrong reading of something the chapter actually says).

The source you must read:

--- source.txt ---
{source}
--- end of source.txt ---

Now call Write with the filled-in content."#
        );
        SessionBuilder::new("write-class-notes-misconceptions", dir)
            .system_prompt(ONE_SHOT_WRITE_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_4_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![(
            "misconceptions_md".into(),
            PathBuf::from(MISCONCEPTIONS_FILENAME),
        )]
    }
}

// ---- Step 3g: assemble-class-notes (deterministic) ------------------------

/// Reads the plan + every per-part file and writes `class-notes.md`. Tolerant
/// of the model wrapping its output in extra prose or stray fences around the
/// per-part section — we extract the meaningful body of each file before
/// concatenating, and trust the plan's concept list as the source of truth
/// for the `### <concept>` headings.
///
/// Matches the heading at any level: `# Worked example`, `## Worked example`,
/// or `### Worked example` are all accepted — observed in real model
/// output. Strips the bullets up to the next markdown heading.
fn extract_section_body(text: &str, heading_name: &str) -> Option<String> {
    let needle = heading_name.to_ascii_lowercase();
    let mut found = false;
    let mut buf = String::new();
    for line in text.lines() {
        let t = line.trim_end();
        if !found {
            let lower = t.trim().to_ascii_lowercase();
            let stripped = lower
                .trim_start_matches('#')
                .trim_start();
            if stripped == needle || stripped.starts_with(&format!("{needle}:")) {
                found = true;
            }
            continue;
        }
        if t.trim_start().starts_with('#') {
            break;
        }
        buf.push_str(t);
        buf.push('\n');
    }
    if found {
        Some(buf.trim_end_matches('\n').to_string())
    } else {
        None
    }
}

/// Pull the `### <name>\n- …` block out of a per-concept file. Tolerates the
/// model picking the wrong heading level (`# Ratios` or `## Ratios` rather
/// than `### Ratios`), mangling casing, or omitting the heading entirely.
/// Also normalizes leading bullets like `-\ ` (an escaped-backslash quirk
/// observed in real model output) back to `- `.
fn extract_concept_block(text: &str, expected_name: &str) -> String {
    fn normalize_bullet(line: &str) -> String {
        let t = line.trim_end();
        let trimmed = t.trim_start();
        // The model sometimes emits `-\ Foo` (literal backslash) instead
        // of `- Foo` — fix it.
        if let Some(rest) = trimmed.strip_prefix("-\\ ") {
            return format!("- {rest}");
        }
        if let Some(rest) = trimmed.strip_prefix("-\\") {
            return format!("- {}", rest.trim_start());
        }
        t.to_string()
    }
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let t = line.trim_start();
        if t.starts_with("# ") || t.starts_with("## ") || t.starts_with("### ") {
            // Take this heading as the concept's heading regardless of level.
            let mut buf = String::new();
            buf.push_str("### ");
            buf.push_str(expected_name);
            buf.push('\n');
            while let Some(peek) = lines.peek() {
                let pt = peek.trim_start();
                if pt.starts_with("### ") || pt.starts_with("## ") || pt.starts_with("# ") {
                    break;
                }
                buf.push_str(&normalize_bullet(peek));
                buf.push('\n');
                lines.next();
            }
            return buf.trim_end_matches('\n').to_string();
        }
    }
    // Fallback: model forgot the heading. Treat everything as the body.
    let mut buf = String::new();
    buf.push_str("### ");
    buf.push_str(expected_name);
    buf.push('\n');
    for line in text.lines() {
        let t = line.trim_end();
        if t.trim_start().starts_with('#') {
            continue;
        }
        buf.push_str(&normalize_bullet(t));
        buf.push('\n');
    }
    buf.trim_end_matches('\n').to_string()
}

struct AssembleClassNotes;
#[async_trait]
impl DeterministicStep for AssembleClassNotes {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let step = "assemble-class-notes";
        let dir = lesson_dir(ctx);
        let plan_text = tokio::fs::read_to_string(dir.join(CLASS_NOTES_PLAN_FILENAME))
            .await
            .map_err(|e| FlowError::Step {
                step: step.into(),
                msg: format!("read {CLASS_NOTES_PLAN_FILENAME}: {e}"),
            })?;
        let plan = parse_class_notes_plan(&plan_text).map_err(|m| FlowError::Step {
            step: step.into(),
            msg: m,
        })?;

        let objectives_text =
            tokio::fs::read_to_string(dir.join(OBJECTIVES_FILENAME)).await.unwrap_or_default();
        let worked_text = tokio::fs::read_to_string(dir.join(WORKED_EXAMPLE_FILENAME))
            .await
            .unwrap_or_default();
        let misc_text = tokio::fs::read_to_string(dir.join(MISCONCEPTIONS_FILENAME))
            .await
            .unwrap_or_default();

        let objectives_body = extract_section_body(&objectives_text, "learning objectives")
            .unwrap_or_else(|| objectives_text.trim().to_string());
        let worked_body = extract_section_body(&worked_text, "worked example")
            .unwrap_or_else(|| worked_text.trim().to_string());
        let misc_body = extract_section_body(&misc_text, "common misconceptions")
            .unwrap_or_else(|| misc_text.trim().to_string());

        let mut concept_blocks: Vec<String> = Vec::new();
        for (i, name) in plan.concepts.iter().enumerate() {
            let n = (i + 1) as u32;
            let path = dir.join(concept_filename(n));
            let body = tokio::fs::read_to_string(&path).await.map_err(|e| FlowError::Step {
                step: step.into(),
                msg: format!("read {}: {e}", path.display()),
            })?;
            concept_blocks.push(extract_concept_block(&body, name));
        }

        let mut out = String::new();
        out.push_str("# ");
        out.push_str(&sanitize_title(&plan.title));
        out.push_str("\n\n## Learning objectives\n");
        out.push_str(objectives_body.trim_end());
        out.push_str("\n\n## Key concepts\n");
        out.push_str(&concept_blocks.join("\n\n"));
        out.push_str("\n\n## Worked example\n");
        out.push_str(worked_body.trim_end());
        out.push_str("\n\n## Common misconceptions\n");
        out.push_str(misc_body.trim_end());
        out.push('\n');

        let path = dir.join(CLASS_NOTES_FILENAME);
        tokio::fs::write(&path, out.as_bytes()).await.map_err(|e| FlowError::Step {
            step: step.into(),
            msg: format!("write {}: {e}", path.display()),
        })?;
        Ok(StepOutcome {
            outputs: vec![(CLASS_NOTES_KEY.into(), path)],
        })
    }
}

// ----- Step 4: write-homework ----------------------------------------------

struct WriteHomework;
impl AgentStepFactory for WriteHomework {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = lesson_dir(ctx);
        let class_notes = std::fs::read_to_string(dir.join(CLASS_NOTES_FILENAME))
            .unwrap_or_else(|_| String::new());
        // Pull just the concept names + a clean title from class-notes.md.
        // Passing the whole assembled file as context confuses Gemma 4 — it
        // reads our title-line placeholder residue (e.g. `# <a short title ...>`)
        // as instructions and shortcuts to "Done." every turn.
        // Strip stray trailing punctuation while keeping original-case
        // display ("Percentages)" → "Percentages").
        let concept_names: Vec<String> = class_notes
            .lines()
            .filter_map(|l| l.trim().strip_prefix("### ").map(|c| {
                c.trim()
                    .trim_end_matches([')', '(', '.', ',', ':', ';', '!', '?'])
                    .trim()
                    .to_string()
            }))
            .filter(|c| !c.is_empty())
            .collect();
        let concept_list = if concept_names.is_empty() {
            "(concepts not parsed from class-notes.md)".into()
        } else {
            concept_names
                .iter()
                .map(|c| format!("  - {c}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let chapter_title = class_notes
            .lines()
            .find_map(|l| l.trim().strip_prefix("# ").map(|t| t.trim().to_string()))
            .filter(|t| !t.contains('<'))
            .unwrap_or_else(|| "today's chapter".into());
        let task = format!(
            r#"You will write the file `{HOMEWORK_FILENAME}` by calling the Write tool. Do NOT reply "Done." before calling Write — the file must exist first.

Write five practice problems for today's homework. The chapter title is: {chapter_title}

Every numbered problem MUST end with ` (maps to: <Concept Name>)` where <Concept Name> is one of these concepts, copied verbatim:

{concept_list}

A downstream validator rejects the file if any numbered line is missing this suffix or names a concept not in the list above. Problems grow in difficulty from 1 to 5.

The file body has this structure, indented here for clarity (the file itself is not indented):

    # Homework — {chapter_title}

    ## Practice problems
    1. <problem statement> (maps to: <concept name>)
    2. <problem statement> (maps to: <concept name>)
    3. <problem statement> (maps to: <concept name>)
    4. <problem statement> (maps to: <concept name>)
    5. <problem statement> (maps to: <concept name>)

    ## Reflection prompt
    <one open-ended question about today's lesson>

    ## Suggested time
    <a realistic number, e.g. "30 minutes">

Replace every <...> placeholder with real content. Do NOT echo the placeholder text.

Now call Write with the filled-in content."#
        );
        SessionBuilder::new("write-homework", dir)
            .system_prompt(ONE_SHOT_WRITE_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_4_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![(HOMEWORK_KEY.into(), PathBuf::from(HOMEWORK_FILENAME))]
    }
}

// ----- Step 5: mk-dir-for-<student> -----------------------------------------

struct MkPerStudentDir {
    slug: String,
}
#[async_trait]
impl DeterministicStep for MkPerStudentDir {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let dest = lesson_dir(ctx).join("per-student").join(&self.slug);
        tokio::fs::create_dir_all(&dest).await.map_err(|e| FlowError::Step {
            step: format!("mk-dir-for-{}", self.slug),
            msg: e.to_string(),
        })?;
        Ok(StepOutcome::default())
    }
}

// ----- Step 6a: plan-tailoring-for-<student> --------------------------------
//
// Splits the old "do everything in one shot" tailor session into a small
// PLAN step (this) and a focused WRITE step (the existing tailor agents,
// which now consume the plan). Per `docs/tailor-decomposition.md`.
//
// The agent emits one structured JSON object that names — for each `###
// <concept>` heading in master class-notes.md — which student interest to
// use and which SPECIFIC element from inside that interest (a character,
// a place, a mechanic, a song, a track, a technique). Also a worked-example
// anchor element, and a per-problem element for the master homework's
// problem count. The output is small (~30 lines of JSON) so the model can
// reason about all the small picks at once without juggling structure.

const TAILORING_PLAN_FILENAME: &str = "tailoring-plan.md";

const PLAN_TAILORING_SYSTEM: &str = r##"You are picking concrete tailoring anchors for one specific student.

You can ONLY use this tool:
  - Write — creates a NEW file inside the working directory.

How to use tools (MANDATORY):
  - Use a `tool_code` fence with a Python-style call:
    ```tool_code
    Write(path="tailoring-plan.md", content="# Plan\n...")
    ```
  - One Write call is enough. After Write succeeds, reply exactly: Done.

You are NOT writing lesson content. You are only picking specific named anchors that a later step will use to write the lesson. Keep the file short — short lines, no prose, no JSON.
"##;

struct PlanTailoring {
    slug: String,
}
impl AgentStepFactory for PlanTailoring {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = lesson_dir(ctx).join("per-student").join(&self.slug);
        let student_root = ctx.root.join("students").join(&self.slug);
        let lesson = lesson_dir(ctx);
        let student_md = std::fs::read_to_string(student_root.join("student.md"))
            .unwrap_or_else(|_| "(student.md not found)".into());
        let tags_json = std::fs::read_to_string(student_root.join("tags.json"))
            .unwrap_or_else(|_| "(tags.json not found)".into());
        let class_notes =
            std::fs::read_to_string(lesson.join(CLASS_NOTES_FILENAME)).unwrap_or_default();
        let master_hw =
            std::fs::read_to_string(lesson.join(HOMEWORK_FILENAME)).unwrap_or_default();
        let concepts: Vec<String> = class_notes
            .lines()
            .filter_map(|l| {
                let t = l.trim();
                t.strip_prefix("### ").map(|c| c.trim().to_string())
            })
            .collect();
        // Pull each master problem's `n. body (maps to: …)` so the planner
        // can craft a scenario whose operand shape matches the operation
        // that specific problem expects.
        let mut master_problem_lines: Vec<(u32, String)> = Vec::new();
        for line in master_hw.lines() {
            let t = line.trim_start();
            let mut chars = t.chars();
            let a = chars.next();
            let b = chars.next();
            let n_str = match (a, b) {
                (Some(x), Some(y)) if x.is_ascii_digit() && (y == '.' || y == ')') => {
                    Some(x.to_string())
                }
                (Some(x), Some(y)) if x.is_ascii_digit() && y.is_ascii_digit() => {
                    if matches!(chars.next(), Some('.') | Some(')')) {
                        Some(format!("{x}{y}"))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(n_str) = n_str {
                if let Ok(n) = n_str.parse::<u32>() {
                    master_problem_lines.push((n, t.to_string()));
                }
            }
        }
        let problem_count = master_problem_lines.len();

        // Markdown template — no nested escaping. Each row is plain text with
        // `key: value` pairs the validator parses with regex.
        let concept_template = concepts
            .iter()
            .map(|c| format!("- concept: {c}\n  interest: <one of the student's tags>\n  named_element: <a specific element from inside that interest>\n  scenario: <a one-line concrete situation from inside that interest that this concept operates on; include real numbers or named entities where possible>"))
            .collect::<Vec<_>>()
            .join("\n");
        // Per-problem template entries quote the master problem so the
        // planner can match the scenario's operands to what the operation
        // expects.
        let problem_template = master_problem_lines
            .iter()
            .map(|(n, body)| {
                format!(
                    "- n: {n}\n  # master problem to mirror: {body}\n  interest: <one of the student's tags>\n  named_element: <a specific element from inside that interest>\n  scenario: <a one-line concrete situation from that interest whose numbers / entities can serve as the operands of the master problem's operation>"
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let _ = problem_count;

        let task = format!(
            r#"You will write the file `{TAILORING_PLAN_FILENAME}` by calling the Write tool. Do NOT reply "Done." before calling Write — the file must exist first.

Pick specific tailoring anchors for this student. The file is plain markdown — no JSON, no code fences inside it.

The file body has this shape (indented here for clarity; the file itself is NOT indented):

    # Tailoring plan

    ## Concepts
    {concept_template}

    ## Worked example
    - interest: <one of the student's tags>
    - named_element: <a specific element from inside that interest>
    - scenario: <a one-line concrete situation from inside that interest that the worked example can operate on; include real numbers or named entities>

    ## Problems
    {problem_template}

Rules:
  - `interest:` is one of the kebab-case tags from `tags.json` below.
  - `named_element:` is a SPECIFIC element from inside that interest — a character, a place, a mechanic, a song, a player, a technique.
  - `scenario:` is the load-bearing field — a CONCRETE micro-situation from that interest that THE PROBLEM'S OPERATION CAN ACT ON. The downstream step uses the scenario's operands (numbers, named entities, quantities) as the operands of the rewritten problem. Shape examples:
      - For a fractions problem with anchor 'Barcelona FC': "Barcelona scored 3 goals out of 8 shots in the first half" — gives the substituter `3` and `8` as numerator/denominator.
      - For a ratios problem with anchor 'Dragon Ball Z': "Goku has a power level of 9,000 while Vegeta has 18,000" — gives the substituter two quantities for a ratio.
      - For a non-math concept with anchor 'Minecraft': "a redstone circuit with 4 pressure plates wired in series" — gives the substituter a named mechanism.
    AVOID generic scenarios like "Goku is fighting" or "Barcelona is playing" — they have no operands.
  - Pick DIFFERENT interests across the concepts when possible.
  - Keep every `concept:` label and every `n:` number from the template. Replace every <...> placeholder with real content.

The student profile and tags you must read:

--- student.md ---
{student_md}
--- end of student.md ---

--- tags.json ---
{tags_json}
--- end of tags.json ---

Now call Write with the filled-in content."#
        );
        SessionBuilder::new(format!("plan-tailoring-for-{}", self.slug), dir)
            .system_prompt(PLAN_TAILORING_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_4_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![(
            format!("tailoring_plan_{}", self.slug),
            PathBuf::from(TAILORING_PLAN_FILENAME),
        )]
    }
}

// ----- Step 6b: validate-tailoring-plan-<slug> (deterministic) --------------

struct ValidateTailoringPlan {
    slug: String,
}

#[async_trait]
impl DeterministicStep for ValidateTailoringPlan {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let step = format!("validate-tailoring-plan-{}", self.slug);
        let path = lesson_dir(ctx)
            .join("per-student")
            .join(&self.slug)
            .join(TAILORING_PLAN_FILENAME);
        let text = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| FlowError::Step {
                step: step.clone(),
                msg: format!("read {}: {e}", path.display()),
            })?;
        let plan = parse_tailoring_plan(&text).map_err(|m| FlowError::Step {
            step: step.clone(),
            msg: m,
        })?;
        if plan.concepts.is_empty() {
            return Err(FlowError::Step {
                step,
                msg: format!("{TAILORING_PLAN_FILENAME} has no concept entries"),
            });
        }
        for (i, c) in plan.concepts.iter().enumerate() {
            for (key, val) in [
                ("concept", &c.concept),
                ("interest", &c.interest),
                ("named_element", &c.named_element),
            ] {
                if val.trim().is_empty() || val.starts_with('<') {
                    return Err(FlowError::Step {
                        step,
                        msg: format!(
                            "{TAILORING_PLAN_FILENAME} concepts[{i}].{key} is empty or echoes the placeholder: '{val}'"
                        ),
                    });
                }
            }
            // Reject only when the named_element echoes the interest tag verbatim
            // ("dinosaurs" for tag "dinosaurs"). A proper-noun single word
            // (e.g. "Allosaurus", "JWST") is legitimately specific.
            if c.named_element.eq_ignore_ascii_case(&c.interest)
                || c.named_element
                    .eq_ignore_ascii_case(&c.interest.replace('-', " "))
            {
                return Err(FlowError::Step {
                    step,
                    msg: format!(
                        "{TAILORING_PLAN_FILENAME} concepts[{i}].named_element ('{}') just echoes the interest tag '{}' — pick a more specific element from inside that interest",
                        c.named_element, c.interest
                    ),
                });
            }
        }
        if plan.worked_example.interest.trim().is_empty()
            || plan.worked_example.named_element.trim().is_empty()
            || plan.worked_example.interest.starts_with('<')
            || plan.worked_example.named_element.starts_with('<')
        {
            return Err(FlowError::Step {
                step,
                msg: format!("{TAILORING_PLAN_FILENAME} worked_example is empty or placeholder"),
            });
        }
        Ok(StepOutcome::default())
    }
}

/// Plain-text plan parsed from the markdown tailoring plan file.
#[derive(Debug, Default)]
pub struct TailoringPlan {
    pub concepts: Vec<TailoringConceptEntry>,
    pub worked_example: TailoringAnchor,
    pub problems: Vec<TailoringProblemEntry>,
}

#[derive(Debug, Default)]
pub struct TailoringConceptEntry {
    pub concept: String,
    pub interest: String,
    pub named_element: String,
    pub scenario: String,
}

#[derive(Debug, Default)]
pub struct TailoringAnchor {
    pub interest: String,
    pub named_element: String,
    pub scenario: String,
}

#[derive(Debug, Default)]
pub struct TailoringProblemEntry {
    pub n: u32,
    pub interest: String,
    pub named_element: String,
    pub scenario: String,
}

/// Parse the markdown plan format:
///
/// ```text
/// ## Concepts
/// - concept: Chloroplasts
///   interest: studio-ghibli
///   named_element: the bathhouse boiler in Spirited Away
///
/// ## Worked example
/// - interest: studio-ghibli
/// - named_element: the camphor tree in Totoro
///
/// ## Problems
/// - n: 1
///   interest: studio-ghibli
///   named_element: Ponyo's underwater kelp
/// ```
///
/// Tolerant of leading list markers, indent variations, and the model
/// dropping the section headings. Section context is tracked so a stray
/// `interest:` line under "## Concepts" knows which concept block it
/// belongs to.
pub fn parse_tailoring_plan(s: &str) -> Result<TailoringPlan, String> {
    enum Section {
        None,
        Concepts,
        WorkedExample,
        Problems,
    }
    let mut sec = Section::None;
    let mut plan = TailoringPlan::default();
    let mut cur_concept: Option<TailoringConceptEntry> = None;
    let mut cur_problem: Option<TailoringProblemEntry> = None;
    let take_value = |line: &str, key: &str| -> Option<String> {
        let t = line.trim().trim_start_matches('-').trim();
        if let Some(rest) = t.strip_prefix(&format!("{key}:")) {
            Some(rest.trim().to_string())
        } else if let Some(rest) = t.strip_prefix(&format!("**{key}**:")) {
            Some(rest.trim().to_string())
        } else {
            None
        }
    };
    for raw_line in s.lines() {
        let line = raw_line.trim_end();
        let lower = line.trim().to_ascii_lowercase();
        if lower.starts_with("## concepts") {
            sec = Section::Concepts;
            continue;
        }
        if lower.starts_with("## worked example") {
            if let Some(c) = cur_concept.take() {
                plan.concepts.push(c);
            }
            sec = Section::WorkedExample;
            continue;
        }
        if lower.starts_with("## problems") {
            if let Some(c) = cur_concept.take() {
                plan.concepts.push(c);
            }
            sec = Section::Problems;
            continue;
        }
        match sec {
            Section::None => {}
            Section::Concepts => {
                if let Some(v) = take_value(line, "concept") {
                    if let Some(c) = cur_concept.take() {
                        plan.concepts.push(c);
                    }
                    let mut c = TailoringConceptEntry::default();
                    c.concept = v;
                    cur_concept = Some(c);
                } else if let Some(v) = take_value(line, "interest") {
                    if let Some(c) = cur_concept.as_mut() {
                        c.interest = v;
                    }
                } else if let Some(v) = take_value(line, "named_element") {
                    if let Some(c) = cur_concept.as_mut() {
                        c.named_element = v;
                    }
                } else if let Some(v) = take_value(line, "scenario") {
                    if let Some(c) = cur_concept.as_mut() {
                        c.scenario = v;
                    }
                }
            }
            Section::WorkedExample => {
                if let Some(v) = take_value(line, "interest") {
                    plan.worked_example.interest = v;
                } else if let Some(v) = take_value(line, "named_element") {
                    plan.worked_example.named_element = v;
                } else if let Some(v) = take_value(line, "scenario") {
                    plan.worked_example.scenario = v;
                }
            }
            Section::Problems => {
                if let Some(v) = take_value(line, "n") {
                    if let Some(p) = cur_problem.take() {
                        plan.problems.push(p);
                    }
                    let mut p = TailoringProblemEntry::default();
                    p.n = v.trim().parse().unwrap_or(0);
                    cur_problem = Some(p);
                } else if let Some(v) = take_value(line, "interest") {
                    if let Some(p) = cur_problem.as_mut() {
                        p.interest = v;
                    }
                } else if let Some(v) = take_value(line, "named_element") {
                    if let Some(p) = cur_problem.as_mut() {
                        p.named_element = v;
                    }
                } else if let Some(v) = take_value(line, "scenario") {
                    if let Some(p) = cur_problem.as_mut() {
                        p.scenario = v;
                    }
                }
            }
        }
    }
    if let Some(c) = cur_concept.take() {
        plan.concepts.push(c);
    }
    if let Some(p) = cur_problem.take() {
        plan.problems.push(p);
    }
    if plan.concepts.is_empty() && plan.problems.is_empty() {
        return Err("tailoring-plan.md has no recognized sections".into());
    }
    Ok(plan)
}

// ----- Step 6c/d: tailor-{notes,hw}-for-<student> (parallel group) ----------

const TAILOR_SYSTEM: &str = r##"You are a careful teaching assistant. Your job is mechanical: take the master file, take the pre-picked tailoring anchors from `tailoring-plan.json`, and produce the per-student file.

You can ONLY use this tool:
  - Write — creates a NEW file inside the working directory.

How to use tools (MANDATORY):
  - Use a `tool_code` fence with a Python-style call:
    ```tool_code
    Write(path="notes.md", content="# Title\n...")
    ```
  - One Write call is enough. After Write succeeds, reply exactly: Done.

You are NOT inventing tailoring anchors here — those have already been chosen for you. Your job is to weave the named anchors into the master's structure. Keep the structure of the master EXACTLY: same headings, same number of concepts, same number of bullets per section. Only the wording of bullets and the worked example change.
"##;

struct TailorHomeworkForStudent {
    slug: String,
}
impl AgentStepFactory for TailorHomeworkForStudent {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = lesson_dir(ctx).join("per-student").join(&self.slug);
        let lesson = lesson_dir(ctx);
        let master_hw = std::fs::read_to_string(lesson.join(HOMEWORK_FILENAME))
            .unwrap_or_else(|_| "(homework.md not found)".into());
        let plan_text =
            std::fs::read_to_string(dir.join(TAILORING_PLAN_FILENAME)).unwrap_or_default();
        let plan = parse_tailoring_plan(&plan_text).unwrap_or_default();

        // Build a deterministic FILL-IN-THE-BLANK template. Each numbered
        // problem becomes a placeholder line: the master problem's text is
        // shown to the model in a small `(master operation: …)` annotation so
        // it knows which OPERATION to apply, then a one-line task tells the
        // model exactly how to fill in the slot using the scenario's
        // operands. There is NO master homework to fall back to — only
        // blanks the model must fill — which is the only reliable way to
        // stop the model from defaulting to verbatim copies.
        let mut master_problems: Vec<(u32, String, String)> = Vec::new();
        for line in master_hw.lines() {
            let t = line.trim_start();
            // Match "1. ..." or "1) ..." with the (maps to: X) suffix.
            let mut chars = t.chars();
            let d1 = chars.next();
            let d2 = chars.next();
            let n_raw = match (d1, d2) {
                (Some(a), Some(b)) if a.is_ascii_digit() && (b == '.' || b == ')') => {
                    Some(a.to_string())
                }
                (Some(a), Some(b)) if a.is_ascii_digit() && b.is_ascii_digit() => {
                    let rest = chars.next();
                    if matches!(rest, Some('.') | Some(')')) {
                        Some(format!("{a}{b}"))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            let Some(n_str) = n_raw else { continue };
            let n: u32 = n_str.parse().unwrap_or(0);
            // Split body from `(maps to: X)` suffix.
            let suffix_idx = match t.rfind("(maps to:") {
                Some(i) => i,
                None => continue,
            };
            // Skip past the number prefix to get the body.
            let after_n = &t[n_str.len()..];
            let after_n = after_n.trim_start_matches([')', '.']).trim_start();
            let body_end = after_n.rfind("(maps to:").unwrap_or(after_n.len());
            let body = after_n[..body_end].trim().to_string();
            let suffix = t[suffix_idx..].to_string();
            master_problems.push((n, body, suffix));
        }

        // Pull the title from the master's first `# …` line so this works
        // for any topic, not just fractions.
        let master_title = master_hw
            .lines()
            .find(|l| l.trim_start().starts_with("# "))
            .map(|l| l.trim().to_string())
            .unwrap_or_else(|| "# Homework".into());
        let mut filled_template = String::new();
        filled_template.push_str(&format!("{master_title}\n\n## Practice problems\n"));
        for (n, body, suffix) in &master_problems {
            let plan_entry = plan.problems.iter().find(|p| p.n == *n);
            let scenario = plan_entry
                .map(|p| {
                    if p.scenario.is_empty() {
                        p.named_element.clone()
                    } else {
                        p.scenario.clone()
                    }
                })
                .unwrap_or_default();
            let interest = plan_entry.map(|p| p.interest.clone()).unwrap_or_default();
            if scenario.is_empty() {
                filled_template.push_str(&format!("{n}. {body} {suffix}\n"));
            } else {
                filled_template.push_str(&format!(
                    "{n}. <one or two sentences. The OPERATION the original problem asked for: \"{body}\". The SCENARIO you must use (from {interest}): \"{scenario}\". Use the scenario's concrete numbers or named entities as the operands the operation acts on. Do NOT use the original problem's numbers; use the scenario's.> {suffix}\n"
                ));
            }
        }
        // Append the trailing sections from the master so the model only
        // worries about the numbered problems.
        let mut tail = String::new();
        let mut in_problems = false;
        for line in master_hw.lines() {
            if line.trim().starts_with("## Reflection") || line.trim().starts_with("## Suggested") {
                in_problems = true;
            }
            if in_problems {
                tail.push_str(line);
                tail.push('\n');
            }
        }
        filled_template.push('\n');
        filled_template.push_str(&tail);

        let task = format!(
            r#"You will write the file `{STUDENT_HW_FILENAME}` by calling the Write tool. Do NOT reply "Done." before calling Write — the file must exist first.

The file content is below as a TEMPLATE with <...> slots on every numbered problem. Your job is to replace every <...> slot with a single concrete 1-2 sentence problem that does exactly what the slot describes — use the scenario's numbers / entities as the operands of the named operation. Leave everything outside the <...> slots unchanged.

The template you must use:

--- template ---
{filled_template}
--- end of template ---

Now call Write with the filled-in content."#
        );
        SessionBuilder::new(format!("tailor-hw-for-{}", self.slug), dir)
            .system_prompt(TAILOR_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_4_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![(
            format!("tailored_hw_{}", self.slug),
            PathBuf::from(STUDENT_HW_FILENAME),
        )]
    }
}

// ----- restore-hw-suffixes-<slug> (deterministic) ---------------------------
//
// The model reliably drops the ` (maps to: <Concept>)` suffix from numbered
// problems even when the prompt told it to preserve them — it treats the
// suffix as decoration when it's busy rewriting the problem body. Rather
// than flood the prompt with reminders, we restore the suffix here
// deterministically from the master homework's suffix-by-n map. If the
// tailored file already has a valid suffix on a line, we leave it alone;
// if it's missing, we append the master's suffix for that problem number.

struct RestoreHomeworkSuffixes {
    slug: String,
}

#[async_trait]
impl DeterministicStep for RestoreHomeworkSuffixes {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let step = format!("restore-hw-suffixes-{}", self.slug);
        let lesson = lesson_dir(ctx);
        let master_path = lesson.join(HOMEWORK_FILENAME);
        let tailored_path = lesson
            .join("per-student")
            .join(&self.slug)
            .join(STUDENT_HW_FILENAME);

        let master = match tokio::fs::read_to_string(&master_path).await {
            Ok(s) => s,
            Err(_) => return Ok(StepOutcome::default()),
        };
        let tailored = match tokio::fs::read_to_string(&tailored_path).await {
            Ok(s) => s,
            Err(_) => return Ok(StepOutcome::default()),
        };

        // Build a `<n> → "(maps to: X)"` map from the master.
        let mut suffix_by_n: std::collections::HashMap<u32, String> = Default::default();
        for line in master.lines() {
            let t = line.trim_start();
            let mut chars = t.chars();
            let d1 = chars.next();
            let d2 = chars.next();
            let n_str = match (d1, d2) {
                (Some(a), Some(b)) if a.is_ascii_digit() && (b == '.' || b == ')') => {
                    Some(a.to_string())
                }
                (Some(a), Some(b)) if a.is_ascii_digit() && b.is_ascii_digit() => {
                    if matches!(chars.next(), Some('.') | Some(')')) {
                        Some(format!("{a}{b}"))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            let Some(n_str) = n_str else { continue };
            let n: u32 = match n_str.parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(idx) = t.rfind("(maps to:") {
                if t[idx..].ends_with(')') {
                    suffix_by_n.insert(n, t[idx..].to_string());
                }
            }
        }
        if suffix_by_n.is_empty() {
            return Ok(StepOutcome::default());
        }

        // Walk the tailored file; for any numbered line that already has a
        // valid suffix leave it alone, otherwise append the master's.
        let mut out = String::with_capacity(tailored.len() + 256);
        let mut patched: u32 = 0;
        for line in tailored.lines() {
            let t = line.trim_end();
            let trimmed = t.trim_start();
            let mut chars = trimmed.chars();
            let d1 = chars.next();
            let d2 = chars.next();
            let n_str = match (d1, d2) {
                (Some(a), Some(b)) if a.is_ascii_digit() && (b == '.' || b == ')') => {
                    Some(a.to_string())
                }
                (Some(a), Some(b)) if a.is_ascii_digit() && b.is_ascii_digit() => {
                    if matches!(chars.next(), Some('.') | Some(')')) {
                        Some(format!("{a}{b}"))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            let n: Option<u32> = n_str.as_ref().and_then(|s| s.parse().ok());
            let has_suffix = trimmed
                .rfind("(maps to:")
                .map(|i| trimmed[i..].ends_with(')'))
                .unwrap_or(false);
            match (n, has_suffix) {
                (Some(n), false) => match suffix_by_n.get(&n) {
                    Some(suffix) => {
                        out.push_str(t);
                        if !t.ends_with(' ') {
                            out.push(' ');
                        }
                        out.push_str(suffix);
                        out.push('\n');
                        patched += 1;
                    }
                    None => {
                        out.push_str(t);
                        out.push('\n');
                    }
                },
                _ => {
                    out.push_str(t);
                    out.push('\n');
                }
            }
        }
        if patched > 0 {
            tokio::fs::write(&tailored_path, out.as_bytes())
                .await
                .map_err(|e| FlowError::Step {
                    step,
                    msg: format!("write {}: {e}", tailored_path.display()),
                })?;
        }
        Ok(StepOutcome::default())
    }
}

// ----- validate-homework-mapping (deterministic) ----------------------------
//
// Enforces the prompt contract: every numbered problem line in the homework
// file must end with ` (maps to: <Concept>)`. Used for the master homework
// and for every per-student tailored homework. We do not try to validate
// that `<Concept>` matches a real `### <concept>` heading from class-notes —
// that's more fragile than useful and the prompt example shows the right
// pattern.

enum HomeworkSource {
    Master,
    PerStudent { slug: String },
}

struct ValidateHomeworkMapping {
    path: String,
    source: HomeworkSource,
}

#[async_trait]
impl DeterministicStep for ValidateHomeworkMapping {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let dir = match &self.source {
            HomeworkSource::Master => lesson_dir(ctx),
            HomeworkSource::PerStudent { slug } => {
                lesson_dir(ctx).join("per-student").join(slug)
            }
        };
        let path = dir.join(&self.path);
        let step_name = match &self.source {
            HomeworkSource::Master => "validate-homework-mapping".to_string(),
            HomeworkSource::PerStudent { slug } => format!("validate-tailored-hw-{slug}"),
        };
        let text = tokio::fs::read_to_string(&path).await.map_err(|e| FlowError::Step {
            step: step_name.clone(),
            msg: format!("read {}: {e}", path.display()),
        })?;

        // Extract the set of valid concept names from the lesson's master
        // class-notes.md (`### <concept>` headings). Tailored homeworks have
        // also seen the master's concept set as their valid set — they're
        // supposed to use the SAME concepts in the SAME order. Without this
        // check the model can swap the entire topic of a tailored homework
        // (observed live: photosynthesis homework rewritten as a stellar
        // evolution homework with fabricated `(maps to: Nebulae)` suffixes).
        let class_notes_path = lesson_dir(ctx).join(CLASS_NOTES_FILENAME);
        let valid_concepts: Vec<String> = match tokio::fs::read_to_string(&class_notes_path).await {
            Ok(s) => s
                .lines()
                .filter_map(|l| {
                    let t = l.trim();
                    t.strip_prefix("### ").map(|c| c.trim().to_string())
                })
                .collect(),
            Err(_) => Vec::new(),
        };
        // Normalize: strip trailing punctuation (`)`, `.`, `,`, `:`, `;`) and
        // lowercase. Live model output sometimes emits stray punctuation on
        // concept names (e.g. `### Percentages)`) which would otherwise cause
        // a false mismatch against the homework's clean `(maps to: Percentages)`.
        let valid_lc: std::collections::HashSet<String> = valid_concepts
            .iter()
            .map(|c| normalize_concept_name(c))
            .collect();

        // Match lines like `1. ...` / `2) ...` and require the literal
        // `(maps to: …)` suffix. Trailing whitespace is tolerated.
        let mut bad: Vec<String> = Vec::new();
        let mut unknown_concept: Vec<String> = Vec::new();
        for line in text.lines() {
            let t = line.trim_end();
            // numbered-problem heuristic: starts with one or two digits then `.` or `)` then space.
            let mut chars = t.chars();
            let d1 = chars.next();
            let d2 = chars.next();
            let is_numbered = match (d1, d2) {
                (Some(a), Some(b)) if a.is_ascii_digit() && (b == '.' || b == ')') => true,
                (Some(a), Some(b)) if a.is_ascii_digit() && b.is_ascii_digit() => {
                    matches!(chars.next(), Some('.') | Some(')'))
                }
                _ => false,
            };
            if !is_numbered {
                continue;
            }
            // Trim trailing markdown emphasis or punctuation we don't care about,
            // then check that the line ends with `(maps to: …)`.
            let suffix_ok = {
                let idx = t.rfind("(maps to:");
                match idx {
                    None => false,
                    Some(i) => t[i..].ends_with(')'),
                }
            };
            if !suffix_ok {
                bad.push(t.to_string());
                continue;
            }
            // Only enforce concept-set membership when we managed to read at
            // least one `### <concept>` heading from class-notes.md. If we
            // didn't, fall back to suffix-only checking so the validator
            // doesn't spuriously fail when class-notes is malformed.
            if !valid_lc.is_empty() {
                let idx = t.rfind("(maps to:").unwrap();
                let inside = t[idx + "(maps to:".len()..t.len() - 1].trim();
                if !valid_lc.contains(&normalize_concept_name(inside)) {
                    unknown_concept.push(format!("'{inside}' on: {t}"));
                }
            }
        }
        if !bad.is_empty() {
            let sample = bad.iter().take(3).cloned().collect::<Vec<_>>().join("\n  • ");
            return Err(FlowError::Step {
                step: step_name,
                msg: format!(
                    "{} numbered problem(s) missing the `(maps to: <Concept>)` suffix. First offending line(s):\n  • {}",
                    bad.len(),
                    sample
                ),
            });
        }
        if !unknown_concept.is_empty() {
            let sample = unknown_concept
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n  • ");
            let known = valid_concepts.join(", ");
            return Err(FlowError::Step {
                step: step_name,
                msg: format!(
                    "{} numbered problem(s) cite a concept that is not in class-notes.md. Known concepts: [{known}]. First offending line(s):\n  • {sample}",
                    unknown_concept.len()
                ),
            });
        }
        Ok(StepOutcome::default())
    }
}

/// Strip `<…>` placeholder residue that the model sometimes echoes verbatim
/// into a title, then collapse leftover whitespace. Without this, Typst
/// rejects the rendered class-notes.md with "unclosed label" because it
/// interprets `<…>` as label syntax. Falls back to a generic placeholder if
/// nothing usable remains.
fn sanitize_title(raw: &str) -> String {
    let mut s = String::with_capacity(raw.len());
    let mut in_angle = false;
    for ch in raw.chars() {
        if ch == '<' {
            in_angle = true;
            continue;
        }
        if ch == '>' {
            in_angle = false;
            continue;
        }
        if !in_angle {
            s.push(ch);
        }
    }
    // Collapse runs of whitespace to a single space, trim surrounding
    // whitespace and stray brackets/punctuation.
    let collapsed: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed
        .trim_matches(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | '[' | ']'));
    if trimmed.is_empty() {
        "Today's lesson".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Strip trailing punctuation and lowercase. Used for tolerant comparison of
/// concept names across the pipeline — live traces show the model occasionally
/// glues a stray `)`, `.`, or `,` onto the end of a `### Concept` heading.
fn normalize_concept_name(s: &str) -> String {
    s.trim()
        .trim_end_matches([')', '(', '.', ',', ':', ';', '!', '?'])
        .trim()
        .to_lowercase()
}

// ----- validate-tailor-divergence (deterministic) ---------------------------
//
// Compares each tailored file against the master on a line-set basis and
// rejects when the student version is too close to the master (i.e. the
// model "tailored" by copying). Catches Gemma's worst tailoring failure
// mode without false positives: a legitimate re-skin preserves headings
// and objectives but rewrites all the example/bullet prose.
struct ValidateTailorDivergence {
    slug: String,
    /// Minimum fraction of lines that must differ between tailored and master.
    /// 0.30 means at least 30 % of the non-empty, non-heading lines must be
    /// new content. Headings count as fixed scaffolding and are excluded.
    min_change_ratio: f32,
}

fn body_lines(s: &str) -> Vec<String> {
    // Normalize whitespace so trivial spacing diffs (e.g. master uses
    // "1.  foo" with two spaces, tailored uses "1. foo" with one) don't
    // make a copy appear different to the HashSet membership check.
    fn normalize(l: &str) -> String {
        l.split_whitespace().collect::<Vec<_>>().join(" ")
    }
    s.lines()
        .map(|l| l.trim().to_string())
        .filter(|l| {
            !l.is_empty()
                && !l.starts_with('#')
                && !l.starts_with("## ")
                && !l.starts_with("### ")
        })
        .map(|l| normalize(&l))
        .collect()
}

#[async_trait]
impl DeterministicStep for ValidateTailorDivergence {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError> {
        let step = format!("validate-tailor-divergence-{}", self.slug);
        let lesson = lesson_dir(ctx);
        let per = lesson.join("per-student").join(&self.slug);
        for (master_name, tailored_name, label) in [
            (HOMEWORK_FILENAME, STUDENT_HW_FILENAME, "homework.md"),
        ] {
            let master = match tokio::fs::read_to_string(lesson.join(master_name)).await {
                Ok(s) => s,
                Err(_) => continue,
            };
            let tailored = match tokio::fs::read_to_string(per.join(tailored_name)).await {
                Ok(s) => s,
                Err(_) => continue,
            };
            let m: std::collections::HashSet<String> = body_lines(&master).into_iter().collect();
            let t: Vec<String> = body_lines(&tailored);
            if t.is_empty() {
                continue;
            }
            let unchanged = t.iter().filter(|l| m.contains(*l)).count();
            let change_ratio = 1.0 - (unchanged as f32 / t.len() as f32);
            if change_ratio < self.min_change_ratio {
                return Err(FlowError::Step {
                    step,
                    msg: format!(
                        "tailored {label} is {:.0}% identical to the master (only {:.0}% of body lines differ; threshold {:.0}%). The model copied instead of translating. Re-run /class-plan to retry.",
                        100.0 * (1.0 - change_ratio),
                        100.0 * change_ratio,
                        100.0 * self.min_change_ratio
                    ),
                });
            }
        }
        Ok(StepOutcome::default())
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
