# Showcase — photosynthesis × Diego

Captured end-to-end on an M-series Mac against the real Gemma 3n E2B Q4_K_M model. The files are numbered in the order the pipeline produces them so you can walk through the data transformation step by step.

## What you are looking at

- `00-teacher-raw-notes.txt` — what the teacher types into the 5-field modal during `/student-add`. Free-text observations across name, age, interests, hobbies, and learning style.
- `02-diego-profile.md` + `02-diego-tags.json` — the output of `/student-add`. The teacher's free-text is digested into a structured `student.md` with mandatory sections and a kebab-case tag list the rest of the system can index against.
- `01-source-chapter.txt` — the public-domain photosynthesis chapter the teacher hands to `/class-plan`. (In the live flow this can also be a PDF that gets OCR'd into the next file.)
- `03-ocr-source.txt` — what the system actually fed to `write-class-notes`. For text-file input this matches `01`; for PDF input this is the Tesseract output.
- `04-master-class-notes.{md,pdf}` — the master class notes, shared across the whole class. Three named concepts, a worked example anchored on a *named tree* pulled from the source, and three common misconceptions a teacher would want to pre-empt.
- `05-master-homework.{md,pdf}` — the master homework. Five problems, each ending with `(maps to: <Concept>)` referencing one of the master's `### <concept>` headings. A deterministic validator confirms every numbered line has the suffix and every concept named is real before the flow proceeds.
- `06-diego-tailoring-plan.md` — the output of the `plan-tailoring-for-diego` agent. The planner reads `02-diego-profile.md` and `02-diego-tags.json` plus the master concept set and picks, for each concept and each homework problem, a *specific element from inside one of Diego's interests* — a locomotive class name, an encyclopedia title, a YouTuber. The planner's job is only this: picking concrete anchors.
- `07-diego-tailored-homework.{md,pdf}` — the output of the `tailor-hw-for-diego` agent. Each problem statement is rewritten around one of the plan's anchors; the master's `(maps to: …)` suffix is preserved verbatim on every line. Diego receives this PDF.

## What to look for

Compare `05-master-homework.md` line-by-line with `07-diego-tailored-homework.md`. The concept mapping is identical; the scenarios are entirely Diego's world.

The `## Notes for tailoring lessons` section in `02-diego-profile.md` is itself an output of an earlier step (the `write-student` agent) and demonstrates the system's operational-tailoring contract — every bullet must name a specific interest AND a specific instructional move. Two bullets later, when the tailoring step picks `GP38-2` and `SD40-2` as anchors, it is pulling those locomotive class names forward from this profile section, which in turn pulled them forward from the teacher's free-text dump.
