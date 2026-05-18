# Tailor decomposition — future work

## Why this exists

`tailor-notes-for-<slug>` and `tailor-hw-for-<slug>` are the two largest
single-shot agent sessions in `/class-plan`. They ask Gemma 4 E2B to
simultaneously:

1. preserve the master's topic, structure, concept headings, and bullet
   counts,
2. invent re-skinned content that uses a specific named element from inside
   one of the student's interests (not just the interest's title),
3. match the student's intellectual register,
4. keep every `(maps to: <Concept>)` suffix intact,
5. avoid both copying the master verbatim and swapping the topic.

In trace recordings against the real model (`traces/phase-2-class-plan-*`),
Gemma 4 collapses into one of two failure modes:

- **Verbatim copy** — output is byte-for-byte the master with whitespace
  edits. (Caught now by `validate-tailor-divergence-<slug>`.)
- **Topic swap** — output abandons photosynthesis and writes a stellar-
  evolution homework with fabricated `(maps to: Nebulae)` suffixes. (Caught
  now by `validate-homework-mapping`'s concept-set check.)

Both failure modes share a root cause: the task is too big for the model's
attention budget. The rest of `/class-plan` and `/student-add` already
follow the project's core rule — *each agent session does exactly one small
thing and the deterministic glue between sessions carries state*. The two
tailor sessions are the outliers.

This document defines the decomposition.

## Principles

- One agent session ↔ one narrow decision. Never multiple decisions per
  session.
- Deterministic steps stitch agent outputs together; agents never assemble
  files larger than their decision scope.
- Each agent session has all its inputs *pre-loaded* in the prompt; we work
  with Gemma's single-turn strength, not against it.
- Validators are the seam between steps; they reject loudly so the failure
  is local, not silent.

## `tailor-notes-for-<slug>` → 4 steps

Current: one agent writes the entire per-student `notes.md`.

Proposed:

### Step A — `plan-tailoring-for-<slug>` *(agent, one-shot Write)*

**Inputs (pre-loaded into prompt)**: `student.md`, `tags.json`, and the
master's headings only (objectives + `### <concept>` names, NOT bullet
content).

**Output**: `tailoring-plan.json` in the per-student dir. Shape:

```json
{
  "concept_plans": [
    {
      "concept": "Chloroplasts",
      "interest": "studio-ghibli",
      "named_element": "the Catbus's grass-lined seats",
      "rewrite_hint": "anchor each bullet in how the seats glow when light hits them"
    },
    {
      "concept": "Chlorophyll",
      "interest": "marine-biology",
      "named_element": "kelp blade chloroplasts seen through a hand lens",
      "rewrite_hint": "tie pigment colour to depth filtering"
    }
  ],
  "worked_example": {
    "interest": "studio-ghibli",
    "named_element": "the camphor tree from My Neighbor Totoro",
    "scenario": "Mei watches sunbeams hit a leaf at noon"
  }
}
```

**Why this is tractable**: the agent only emits one small JSON object —
no prose. The decision is "pick one named element per concept." Gemma is
strong at small structured tasks.

### Step B — `validate-tailoring-plan-<slug>` *(deterministic)*

Parses the JSON; verifies every `concept` matches a master `### <concept>`
heading; verifies `named_element` is at least 3 words and not empty; counts
`concept_plans` matches concept count.

### Step C — `tailor-concept-<slug>-<concept-slug>` *(agent, one-shot Write, per concept)*

**Inputs**: one master concept block + the corresponding `concept_plan`
entry + the student's `student.md` (for register).

**Task**: emit ONLY the rewritten bullets for this one concept block — to
`concepts/<concept-slug>.md` in the per-student dir.

**Why this is tractable**: the agent rewrites ~3–5 bullets, knowing exactly
which named element to use. No structural decisions, no other concepts in
view.

### Step D — `tailor-worked-example-<slug>` *(agent, one-shot Write)*

**Inputs**: the master's `## Worked example` + the plan's
`worked_example` entry + `student.md`.

**Task**: write the rewritten worked example to `worked-example.md`. One
paragraph. Tiny scope.

### Step E — `assemble-notes-<slug>` *(deterministic)*

Reads:
- the master `class-notes.md` (for `## Learning objectives`,
  `## Common misconceptions`, headings),
- each `concepts/<concept-slug>.md`,
- `worked-example.md`.

Writes `notes.md` by template-substituting the rewritten blocks under their
original headings. Adds one student-specific misconception bullet only if
the plan declared one. Deterministic glue; no model invocation.

### Step F — `validate-tailor-divergence-<slug>` *(already implemented)*

Already in place. After the new decomposition it should rarely fail because
each concept's rewrite was forced by Step C.

## `tailor-hw-for-<slug>` → 3 steps

Current: one agent writes the entire per-student `homework.md`.

Proposed:

### Step A — `plan-hw-tailoring-for-<slug>` *(agent, one-shot Write)*

**Inputs**: `student.md`, `tags.json`, and the master homework's *suffixes
only* (e.g. "5 problems, mapping to concepts: Chloroplasts, Chlorophyll,
The Light Reaction and Calvin Cycle, …").

**Output**: `hw-plan.json` listing per-problem: which interest, which
named element, one-sentence framing.

### Step B — `tailor-problem-<slug>-<n>` *(agent, one-shot Write, per problem)*

**Inputs**: one master problem + the corresponding `hw-plan.json` entry +
`student.md`.

**Task**: rewrite this one problem statement and append the verbatim
`(maps to: <Concept>)` suffix from the master. Write to `problems/<n>.md`.

**Why this is tractable**: the agent rewrites one sentence. The concept
suffix is supplied as a fixed string; the agent's job is just to embed the
named element.

### Step C — `assemble-hw-<slug>` *(deterministic)*

Concatenates `problems/1.md` through `problems/<n>.md` under
`## Practice problems`. Copies `## Reflection prompt` and `## Suggested
time` verbatim from the master (these don't need to be tailored). Writes
`homework.md`.

Existing `validate-homework-mapping` and
`validate-tailor-divergence-<slug>` run after.

## Parallelism

Step C of notes (per-concept tailor) and Step B of homework (per-problem
tailor) are independent and can run in the `tailor` parallel group bounded
by `GEMMA_TEACH_PARALLELISM`. The current single-session shape already
runs `tailor-notes` and `tailor-hw` in parallel for each student; the new
shape multiplies the parallelism by `concept_count` (3–5) and
`problem_count` (5), which is fine on M-series and serializes cleanly on
older hardware.

## Cost note

The current single-session tailor uses ~440 tokens × 2 = ~880 tokens per
student. The decomposed flow does ~150 (plan) + ~3×80 (per concept) +
~120 (worked example) + ~150 (hw plan) + ~5×60 (per problem) ≈ 960
tokens per student. Cost is roughly flat. **Latency** is slightly higher
because of more turns, but each turn is shorter and runs in parallel —
expect ~equal wall time on M-series.

## Migration path

1. Land Step A + Step B for both files (plan + validator), keeping the
   existing big-tailor as a fallback.
2. Land Step C (per-concept / per-problem agents) behind a feature flag
   `GEMMA_TEACH_DECOMPOSED_TAILOR=1`.
3. Run trace battery on all 5 fixture students; compare divergence ratios
   and concept-coverage to the single-session baseline.
4. Flip the flag default once decomposed beats baseline on every fixture.
5. Remove the big-tailor session and its prompts.

## Why we shipped Phase 2 without this

This decomposition is a real refactor — 6–7 new step types, new artifact
schema (`tailoring-plan.json`, per-concept files), and a fresh trace
battery to tune the planner agent. Phase 2's scope was hardening +
prompt iteration. Doing the decomposition in the same iteration would
have buried the parser fixes, validator additions, and modal expansion
under structural change. The two new validators
(`validate-homework-mapping` concept-set check and
`validate-tailor-divergence`) make the current single-session tailor
*safe* — it can fail loudly instead of silently producing copies. That
makes the current shape acceptable for a v1, and gives Phase 3 a clean
diff to operate on.
