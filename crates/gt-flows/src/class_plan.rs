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
        StepNode::agent("write-class-notes", WriteClassNotes),
        StepNode::agent("write-homework", WriteHomework),
        StepNode::det(
            "validate-homework-mapping",
            ValidateHomeworkMapping {
                path: HOMEWORK_FILENAME.into(),
                source: HomeworkSource::Master,
            },
        ),
    ];
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

// ----- Step 3: write-class-notes -------------------------------------------

// One-shot pattern: each session gets its inputs pre-loaded into the task
// prompt and only needs to emit ONE Write tool call. This works with Gemma 3n's
// single-turn strength instead of fighting it through multi-step Read+Write.
const ONE_SHOT_WRITE_SYSTEM: &str = r##"You are a careful teaching assistant working inside a fixed working directory.

You can ONLY use this tool:
  - Write — creates a NEW file inside the working directory.

How to use tools:
  - Use `tool_code` fences to call tools, e.g.:
    ```tool_code
    Write(path="class-notes.md", content="# Title\n...")
    ```
  - One Write call is enough for this task. After Write succeeds, reply exactly: Done.
"##;

struct WriteClassNotes;
impl AgentStepFactory for WriteClassNotes {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = lesson_dir(ctx);
        let source = std::fs::read_to_string(dir.join(SOURCE_TXT_FILENAME))
            .unwrap_or_else(|_| "(source.txt not found)".into());
        let task = format!(
            r#"Below is the OCR'd content of a textbook chapter. Write `{CLASS_NOTES_FILENAME}` with EXACTLY this structure:

# <title — infer from the source>

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
- A single concrete example that uses these concepts. Pull a NAMED entity or a numerical value directly from the source — do not paraphrase. The example must reference at least two of the Key concepts by name.

## Common misconceptions
- 2-4 bullets a teacher should pre-empt.

Stay strictly faithful to the source. Do not introduce material that is not in the source.
After Write succeeds, reply: Done.

--- source.txt ---
{source}
--- end of source.txt ---"#
        );
        SessionBuilder::new("write-class-notes", dir)
            .system_prompt(ONE_SHOT_WRITE_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_3n_e2b())
    }
    fn output_keys(&self) -> Vec<(String, PathBuf)> {
        vec![(CLASS_NOTES_KEY.into(), PathBuf::from(CLASS_NOTES_FILENAME))]
    }
}

// ----- Step 4: write-homework ----------------------------------------------

struct WriteHomework;
impl AgentStepFactory for WriteHomework {
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder {
        let dir = lesson_dir(ctx);
        let class_notes = std::fs::read_to_string(dir.join(CLASS_NOTES_FILENAME))
            .unwrap_or_else(|_| "(class-notes.md not found)".into());
        let task = format!(
            r#"Write today's homework based on the class-notes below.

Constraints (you, the model, must follow these — they describe how to fill in the template, do NOT include this paragraph in the file):
  • Every numbered problem MUST end with ` (maps to: <Concept Name>)`. `<Concept Name>` is one of the `### <concept>` headings from class-notes.md, copied verbatim. A downstream validator rejects the file if any numbered line is missing this suffix.
  • Problems grow in difficulty from 1 to 5.
  • Replace every `<…>` placeholder below with real content. Do NOT echo the placeholder text.

Write `{HOMEWORK_FILENAME}` with EXACTLY this structure:

```
# Homework — <same title as class-notes.md>

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
```

After Write succeeds, reply: Done.

--- class-notes.md ---
{class_notes}
--- end of class-notes.md ---"#
        );
        SessionBuilder::new("write-homework", dir)
            .system_prompt(ONE_SHOT_WRITE_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_3n_e2b())
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
            r#"Pick specific tailoring anchors for this student. Write `{TAILORING_PLAN_FILENAME}` with EXACTLY this shape (plain markdown, no JSON, no code fences inside the file):

```
# Tailoring plan

## Concepts
{concept_template}

## Worked example
- interest: <one of the student's tags>
- named_element: <a specific element from inside that interest>
- scenario: <a one-line concrete situation from inside that interest that the worked example can operate on; include real numbers or named entities>

## Problems
{problem_template}
```

Rules:
  • `interest:` is one of the kebab-case tags from `tags.json` below.
  • `named_element:` is a SPECIFIC element from inside that interest — a character, a place, a mechanic, a song, a player, a technique.
  • `scenario:` is the load-bearing field — it must be a CONCRETE micro-situation from that interest that THE PROBLEM'S OPERATION CAN ACT ON. The downstream step uses the scenario's operands (numbers, named entities, quantities) as the operands of the rewritten problem. Examples of the shape we want:
      – For a fractions problem with anchor 'Barcelona FC': "Barcelona scored 3 goals out of 8 shots in the first half" — gives the substituter `3` and `8` to use as numerator/denominator.
      – For a ratios problem with anchor 'Dragon Ball Z': "Goku has a power level of 9,000 while Vegeta has 18,000" — gives the substituter the two quantities for a ratio.
      – For a non-math concept (a process or definition) with anchor 'Minecraft': "a redstone circuit with 4 pressure plates wired in series" — gives the substituter a named mechanism.
    AVOID generic scenarios like "Goku is fighting" or "Barcelona is playing" — they have no operands.
  • Pick DIFFERENT interests across the concepts when possible.
  • Keep every `concept:` label and every `n:` number from the template above. Replace every `<…>` placeholder with real content.

--- student.md ---
{student_md}
--- end of student.md ---

--- tags.json ---
{tags_json}
--- end of tags.json ---

After Write succeeds, reply: Done."#
        );
        SessionBuilder::new(format!("plan-tailoring-for-{}", self.slug), dir)
            .system_prompt(PLAN_TAILORING_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_3n_e2b())
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
        // stop Gemma 3n from defaulting to verbatim copies.
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
            r#"Write `{STUDENT_HW_FILENAME}`. The file content is below as a TEMPLATE with `<…>` slots on every numbered problem. Your job is to replace every `<…>` slot with a single concrete 1–2 sentence problem that does exactly what the slot describes — use the scenario's numbers / entities as the operands of the named operation. Leave everything outside the `<…>` slots unchanged.

--- template ---
{filled_template}
--- end of template ---

After Write succeeds, reply: Done."#
        );
        SessionBuilder::new(format!("tailor-hw-for-{}", self.slug), dir)
            .system_prompt(TAILOR_SYSTEM)
            .task_prompt(task)
            .allowed_tools(["Write"])
            .model_profile(gt_core::ModelProfile::gemma_3n_e2b())
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
// Gemma 3n reliably drops the ` (maps to: <Concept>)` suffix from numbered
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
        let valid_lc: std::collections::HashSet<String> =
            valid_concepts.iter().map(|c| c.to_lowercase()).collect();

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
                if !valid_lc.contains(&inside.to_lowercase()) {
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
