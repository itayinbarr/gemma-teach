# Showcase — fractions × Diego

Captured end-to-end on an M-series Mac against the real Gemma 3n E2B Q4_K_M model. The files are numbered in the order the pipeline produces them so you can walk through the data transformation step by step.

The showcase pairs a math chapter on fractions and ratios with a 6th-grade Barcelona-FC obsessed student. Math is the cleanest domain to demonstrate the system because mathematical operations have explicit operands; the per-student tailoring uses real numbers and named entities from the student's world as those operands, not as decorative settings.

## Files

- `00-teacher-raw-notes.txt` — what the teacher types into the 5-field modal during `/student-add`. Free-text observations across name, age, interests, hobbies, and learning style.
- `02-diego-profile.md` + `02-diego-tags.json` — the output of `/student-add`. The teacher's free-text is digested into a structured `student.md` with mandatory sections and a kebab-case tag list the rest of the system can index against.
- `01-source-chapter.txt` — the math chapter the teacher hands to `/class-plan`. (In the live flow this can also be a PDF that gets OCR'd into the next file.)
- `03-ocr-source.txt` — what the system actually fed to `write-class-notes`.
- `04-master-class-notes.{md,pdf}` — the master class notes, shared across the whole class. Three named concepts (Equivalent Fractions, Ratios, Fractions) plus a worked example and a misconceptions list.
- `05-master-homework.{md,pdf}` — the master homework. Five problems, each ending with `(maps to: <Concept>)` referencing one of the master's `### <concept>` headings.
- `06-diego-tailoring-plan.md` — the output of the `plan-tailoring-for-diego` agent. The planner reads `02-diego-profile.md` and `02-diego-tags.json` plus each master problem's text, and picks per-concept and per-problem a *scenario* from inside one of Diego's interests whose concrete numbers can serve as the operands of the master problem's operation. For problem 1 (find an equivalent fraction for 2/3), it picks *"Barcelona scored 2 goals out of 3 shots in the first half"* — the numerator and denominator are now soccer stats. For problem 2 (simplify a ratio), it picks two Dragon Ball Z power levels — 4,000 vs 18,000 — whose ratio Diego can compute.
- `07-diego-tailored-homework.{md,pdf}` — the output of the `tailor-hw-for-diego` agent. The harness deterministically builds a fill-in-the-blank template from the plan, the agent fills the blanks, and a deterministic restore-suffixes step appends any `(maps to: …)` suffix the model dropped during rewriting. Diego receives this PDF.

## What to look for

Compare `05-master-homework.md` line-by-line with `07-diego-tailored-homework.md`. The concept-mapping suffix is identical on every line; the problem statements are reformulated so the named-interest scenarios provide the math operands.

The data flow from Diego's free-text notes (`00`) to the soccer-themed problem 1 (`07`) is the load-bearing demonstration: the teacher mentions Barcelona scoring stats in `00`, that detail surfaces verbatim into the `## Notes for tailoring lessons` section of `02-diego-profile.md`, the planner pulls it forward into the per-problem `scenario:` field of `06-diego-tailoring-plan.md`, and the substitution agent uses those exact numbers as the fraction's numerator and denominator in `07-diego-tailored-homework.md`. Nothing in this chain leaves the laptop.
