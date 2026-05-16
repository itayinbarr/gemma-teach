//! Battery test: drive `/student-add` through five elaborate fixture inputs
//! against MockBackend. The fixtures live under
//! `tests/fixtures/students/*.txt` and represent the kind of input a teacher
//! would now produce with the 5-field modal (vs the old 2-field freeform).
//!
//! The mock supplies a high-quality `student.md` + `tags.json` per fixture so
//! this test locks in the *contract* (the prompt accepts richer input and
//! the flow runs cleanly), not the model's actual generation quality. The
//! companion smoke test (gated on `GEMMA_TEACH_SMOKE=1`) drives the same
//! fixtures through the real model into a tmpdir for human inspection.

use chrono::NaiveDate;
use gt_core::backend::{MockBackend, MockScript, StopReason};
use gt_core::tool::ToolRegistry;
use gt_flows::orchestrator::Orchestrator;
use gt_flows::student_add::flow_with_ctx;
use std::sync::Arc;
use tempfile::tempdir;

struct Fixture {
    name: &'static str,
    slug: &'static str,
    txt: &'static str,
    expected_tags: &'static [&'static str],
    student_md: &'static str,
    tags_json: &'static str,
}

const ELARA: Fixture = Fixture {
    name: "Elara",
    slug: "elara",
    txt: include_str!("fixtures/students/elara-rich.txt"),
    expected_tags: &["astrophysics", "magic-the-gathering", "hard-sci-fi"],
    student_md: "# Elara\n\n## Snapshot\n- 14, 9th grade, perfectionist with deep astrophysics fluency.\n- Competitive Magic: The Gathering player.\n\n## Interests\n- Astrophysics, JWST press conferences, HR diagram\n- Exoplanet atmospheres\n- Hard sci-fi (Ted Chiang, Cixin Liu, Greg Egan)\n\n## Hobbies\n- Stargazing with 8\" Dobsonian\n- MTG deckbuilding (Pioneer, Rakdos Midrange)\n- Watching Kurzgesagt, PBS Space Time\n\n## Media they love\n- Three-Body trilogy\n- Outer Wilds, No Man's Sky\n- LSV / Reid Duke MTG content\n\n## Notes for tailoring lessons\n- Frame open-ended prompts as bounded puzzles with known constraints — she freezes on truly open prompts.\n- Use MTG deckbuilding as a metaphor for hypothesis revision: each card is a variable.\n- Ask her to predict before deriving; she over-explains otherwise.\n- Watch for quiet 'I'm fine' when stuck — she won't ask for help unprompted.\n",
    tags_json: "[\"astrophysics\",\"jwst\",\"magic-the-gathering\",\"hard-sci-fi\",\"stargazing\",\"outer-wilds\"]",
};

const DIEGO: Fixture = Fixture {
    name: "Diego",
    slug: "diego",
    txt: include_str!("fixtures/students/diego-rich.txt"),
    expected_tags: &["trains", "dinosaurs"],
    student_md: "# Diego\n\n## Snapshot\n- 9, 4th grade, fidgety but precise when teaching back.\n- Twin obsessions: trains and theropod dinosaurs.\n\n## Interests\n- North American Class I railroads; narrow-gauge logging\n- Theropod dinosaurs (Allosaurus, Spinosaurus)\n- HO-scale model railroading\n\n## Hobbies\n- Building HO-scale layouts with cardboard mountains\n- Reading DK Smithsonian 'Dinosaurs!' end-to-end\n- Playing Train Simulator Classic\n\n## Media they love\n- Magic Tree House dinosaur books\n- Mark Felton railroad shorts\n- Trains magazine\n\n## Notes for tailoring lessons\n- Use timed mini-tasks (3 sentences in 5 minutes) — long writing collapses by sentence three.\n- Ask him to teach the concept back to his younger sister; his explanations are precise.\n- Avoid sarcasm — he reads it literally.\n- Praise specific moves, not intelligence.\n",
    tags_json: "[\"trains\",\"dinosaurs\",\"model-railroading\",\"theropods\",\"magic-tree-house\"]",
};

const AMAYA: Fixture = Fixture {
    name: "Amaya",
    slug: "amaya",
    txt: include_str!("fixtures/students/amaya-rich.txt"),
    expected_tags: &["ballet", "houseplants"],
    student_md: "# Amaya\n\n## Snapshot\n- 13, 7th grade, serious ballet dancer at pre-pro level.\n- Houseplant propagator, slow deep reader.\n\n## Interests\n- Vaganova ballet (level 4)\n- Plant propagation (Monstera adansonii, pothos, ZZ)\n- Sandra Cisneros short stories; A Wrinkle in Time\n\n## Hobbies\n- Daily ballet plus Pilates\n- Tending ~30 houseplants\n- Listening to Ludovico Einaudi / Joe Hisaishi while working\n\n## Media they love\n- Misty Copeland interviews\n- Crash Course Botany\n- Anne with an E\n\n## Notes for tailoring lessons\n- Hand her hard questions ahead of time; her overnight contributions are the strongest.\n- Avoid cold-calling — shoulders rise and eye contact drops.\n- Let her connect abstract ideas to embodied movement (port de bras for gradients).\n- Give her ~30 seconds to transition between tasks.\n",
    tags_json: "[\"ballet\",\"vaganova\",\"houseplants\",\"propagation\",\"film-scores\",\"sandra-cisneros\"]",
};

const NOOR: Fixture = Fixture {
    name: "Noor",
    slug: "noor",
    txt: include_str!("fixtures/students/noor-rich.txt"),
    expected_tags: &["robotics", "badminton"],
    student_md: "# Noor\n\n## Snapshot\n- 11, 6th grade, robotics club lead.\n- ELL learner (Urdu at home), badminton competitor.\n\n## Interests\n- Arduino-based robotics (line-followers, sensors, servos)\n- Badminton (under-12 singles, ranked 4th)\n- Mechatronics tutorials\n\n## Hobbies\n- Building Arduino projects (temperature logger that texts her dad)\n- Badminton training and tournaments\n- Roblox over video call with cousins in Lahore\n\n## Media they love\n- How To Mechatronics YouTube\n- The Karate Kid\n- Roblox\n\n## Notes for tailoring lessons\n- Pre-teach unit vocabulary; she sometimes substitutes the wrong technical term mid-sentence.\n- Pair every worked example with a 'doesn't apply because…' non-example — she over-applies.\n- Praise her method on wrong answers, not just correct ones — performance anxiety from parental expectations.\n- Use her line-follower as the canonical state-machine analogy.\n",
    tags_json: "[\"arduino\",\"robotics\",\"line-follower\",\"badminton\",\"mechatronics\",\"roblox\"]",
};

const SAM: Fixture = Fixture {
    name: "Sam",
    slug: "sam",
    txt: include_str!("fixtures/students/sam-rich.txt"),
    expected_tags: &["fl-studio", "k-pop"],
    student_md: "# Sam\n\n## Snapshot\n- 14, 9th grade, self-taught FL Studio producer.\n- Low math confidence after failing the fractions unit (41%).\n\n## Interests\n- Hip-hop and K-pop beatmaking in FL Studio\n- 808s, sub-bass, sidechain compression\n- New Jeans / late J-Dilla 'swing' feel\n\n## Hobbies\n- FL Studio every night for 45+ minutes\n- Sharing loops on a private Discord with three friends\n- Beat Saber when the headset is free\n\n## Media they love\n- NewJeans, IVE, Stray Kids, SHINee, BIGBANG\n- Song Exploder podcast\n- DKDKTV K-pop reaction channel\n\n## Notes for tailoring lessons\n- Re-skin fraction problems as beat subdivisions and time signatures; he is already fluent verbally.\n- Avoid public correction — he deflects with laughter and disengages.\n- Wait through the laugh, then redirect one-on-one.\n- Frame success as 'you already do this every night' — he needs to feel smart again.\n",
    tags_json: "[\"fl-studio\",\"beatmaking\",\"k-pop\",\"newjeans\",\"j-dilla\",\"hip-hop\"]",
};

const BATTERY: &[Fixture] = &[ELARA, DIEGO, AMAYA, NOOR, SAM];

#[tokio::test]
async fn student_battery_all_five_fixtures_run_clean() {
    let dir = tempdir().unwrap();
    let root = dir.path().to_path_buf();

    for f in BATTERY {
        run_one(&root, f).await;
        assert_one_on_disk(&root, f).await;
    }

    // Sanity: every student dir exists and has tags.json, intersections.json.
    for f in BATTERY {
        let s = root.join("students").join(f.slug);
        assert!(s.join("student.md").exists(), "{}: student.md", f.slug);
        assert!(s.join("tags.json").exists(), "{}: tags.json", f.slug);
        assert!(
            s.join("intersections.json").exists(),
            "{}: intersections.json",
            f.slug
        );
    }
}

async fn run_one(root: &std::path::Path, f: &Fixture) {
    let backend = Arc::new(MockBackend::new());

    // write-student: emit a high-quality student.md.
    backend.push(
        MockScript::new()
            .tool(
                "Write",
                serde_json::json!({"path": "student.md", "content": f.student_md}),
            )
            .done(StopReason::Eos),
    );
    backend.push(MockScript::new().text("Done.").done(StopReason::Eos));

    // extract-tags one-shot.
    backend.push(
        MockScript::new()
            .text("Done.")
            .tool(
                "Write",
                serde_json::json!({"path": "tags.json", "content": f.tags_json}),
            )
            .done(StopReason::Eos),
    );

    let tools = ToolRegistry::new()
        .register(Arc::new(gt_tools::ReadTool))
        .register(Arc::new(gt_tools::WriteTool))
        .register(Arc::new(gt_tools::EditTool));

    let (flow, ctx) = flow_with_ctx(
        root.to_path_buf(),
        NaiveDate::from_ymd_opt(2026, 5, 15).unwrap(),
        f.name.into(),
        f.txt.into(),
    );

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
    let _ = handle.join.await.expect("join").expect("flow ok");
    flow_drain.await.unwrap();
}

async fn assert_one_on_disk(root: &std::path::Path, f: &Fixture) {
    let s = root.join("students").join(f.slug);
    let md = tokio::fs::read_to_string(s.join("student.md")).await.unwrap();
    assert!(md.starts_with(&format!("# {}", f.name)), "{}: bad heading", f.slug);
    assert!(md.contains("## Notes for tailoring lessons"), "{}: missing tailoring section", f.slug);
    let tags: Vec<String> =
        serde_json::from_str(&tokio::fs::read_to_string(s.join("tags.json")).await.unwrap())
            .unwrap();
    for needle in f.expected_tags {
        assert!(
            tags.iter().any(|t| t == needle),
            "{}: tags.json missing '{}': {:?}",
            f.slug,
            needle,
            tags
        );
    }
}

#[test]
fn fixture_files_are_substantive() {
    // Every fixture must be long enough to be richer than the old 2-line
    // freeform description — guards against accidentally shrinking the
    // battery later.
    for f in BATTERY {
        assert!(
            f.txt.len() > 800,
            "{} fixture is only {} chars — must be a substantial teacher dump",
            f.slug,
            f.txt.len()
        );
    }
}
