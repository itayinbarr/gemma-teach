//! TUI application state.

use chrono::Local;
use gt_core::backend::LlmBackend;
use gt_core::ids::StepId;
use gt_core::session_event::{FlowEvent, SessionEvent, StepDescriptor, StepState};
use gt_core::tool::ToolRegistry;
use gt_flows::class_plan::flow_with_ctx as class_plan_flow;
use gt_flows::orchestrator::{Orchestrator, OrchestratorHandle};
use gt_flows::student_add::flow_with_ctx as student_add_flow;
use gt_flows::student_edit::flow_with_ctx as student_edit_flow;
use gt_tools::{TesseractRunner, TypstRunner};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct StepRow {
    pub id: StepId,
    pub name: String,
    pub kind: String,
    pub state: StepState,
    pub streaming_text: String,
    pub last_tool: Option<String>,
    pub artifacts: Vec<String>,
}

#[derive(Debug)]
pub enum AppMode {
    Idle,
    StudentAddModal(StudentAddForm),
    StudentEditModal(StudentEditForm),
    FlowActive,
    Help,
}

#[derive(Debug, Default)]
pub struct StudentAddForm {
    pub name: String,
    pub description: String,
    pub focus: FormField,
}

#[derive(Debug, Default)]
pub struct StudentEditForm {
    pub name: String,
    pub notes: String,
}

pub fn slug_or_self(s: &str) -> String {
    // Lowercase, kebab-case, matching student_add's slugify.
    let mut out = String::with_capacity(s.len());
    let mut last = false;
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last = false;
        } else if !last && !out.is_empty() {
            out.push('-');
            last = true;
        }
    }
    if out.ends_with('-') {
        out.pop();
    }
    out
}

#[derive(Debug, Default, Clone, Copy)]
pub enum FormField {
    #[default]
    Name,
    Description,
}

pub struct App {
    pub root: PathBuf,
    pub backend: Arc<dyn LlmBackend>,
    pub mode: AppMode,
    pub input: String,
    pub messages: VecDeque<String>, // header + slash output
    pub steps: Vec<StepRow>,
    pub selected_step: usize,
    pub flow_name: Option<String>,
    pub flow_started_at: Option<std::time::Instant>,
    pub tick_count: u64,
    pub running_handle: Option<RunningFlow>,
    pub should_quit: bool,
}

pub struct RunningFlow {
    pub flow_rx: mpsc::Receiver<FlowEvent>,
    pub session_rxs: HashMap<StepId, mpsc::Receiver<SessionEvent>>,
    pub _join: tokio::task::JoinHandle<Result<gt_flows::FlowCtx, gt_flows::step::FlowError>>,
}

impl App {
    pub fn new(root: PathBuf, backend: Arc<dyn LlmBackend>) -> Self {
        let mut messages = VecDeque::new();
        messages.push_back(format!(
            "Notebook: {}   Type a / command (/student-add, /class-plan <pdf>, /student-edit <name>, /help, /quit)",
            root.display()
        ));
        Self {
            root,
            backend,
            mode: AppMode::Idle,
            input: String::new(),
            messages,
            steps: Vec::new(),
            selected_step: 0,
            flow_name: None,
            flow_started_at: None,
            tick_count: 0,
            running_handle: None,
            should_quit: false,
        }
    }

    pub fn log(&mut self, line: impl Into<String>) {
        self.messages.push_back(line.into());
        while self.messages.len() > 200 {
            self.messages.pop_front();
        }
    }

    pub fn current_step(&self) -> Option<&StepRow> {
        self.steps.get(self.selected_step)
    }

    /// Drain whatever flow / session events are currently buffered without
    /// blocking. Called every tick by the UI loop.
    pub fn drain_events(&mut self) {
        if let Some(rf) = self.running_handle.as_mut() {
            while let Ok(ev) = rf.flow_rx.try_recv() {
                handle_flow_event(&mut self.steps, ev, &mut self.flow_name);
            }
            let step_ids: Vec<StepId> = rf.session_rxs.keys().copied().collect();
            for id in step_ids {
                if let Some(rx) = rf.session_rxs.get_mut(&id) {
                    while let Ok(ev) = rx.try_recv() {
                        if let Some(row) = self.steps.iter_mut().find(|s| s.id == id) {
                            handle_session_event(row, ev);
                        }
                    }
                }
            }
            // If the flow is done (all steps Done/Failed and the channel closed)
            // we can clear the running handle so a new flow can start.
            let all_done = !self.steps.is_empty()
                && self.steps.iter().all(|s| {
                    matches!(s.state, StepState::Done | StepState::Failed)
                });
            if all_done {
                self.log("Flow complete.");
                self.running_handle = None;
                self.mode = AppMode::Idle;
            }
        }
    }

    fn default_registry() -> ToolRegistry {
        ToolRegistry::new()
            .register(Arc::new(gt_tools::ReadTool))
            .register(Arc::new(gt_tools::WriteTool))
            .register(Arc::new(gt_tools::EditTool))
    }

    fn templates_dir() -> PathBuf {
        // Resolve against the binary's parent directory if the build artifact
        // sits under target/, else fall back to the repo's templates dir.
        let here = std::env::current_exe().ok();
        if let Some(exe) = here {
            // try ../../../templates/typst from target/<profile>/<bin>
            if let Some(repo) = exe.parent().and_then(|p| p.parent()).and_then(|p| p.parent()) {
                let cand = repo.join("templates/typst");
                if cand.exists() {
                    return cand;
                }
            }
        }
        PathBuf::from("templates/typst")
    }

    pub fn start_student_add(&mut self, name: String, description: String) {
        let date = Local::now().date_naive();
        let (flow, ctx) = student_add_flow(self.root.clone(), date, name.clone(), description);
        self.kick_off("/student-add", format!("Starting /student-add for {name}…"), flow, ctx);
    }

    pub fn start_class_plan(&mut self, pdf: PathBuf) {
        let date = Local::now().date_naive();
        let ocr = Arc::new(TesseractRunner::new());
        let pdfr = Arc::new(TypstRunner::new());
        let templates = Self::templates_dir();
        match class_plan_flow(self.root.clone(), date, pdf.clone(), ocr, pdfr, templates) {
            Ok((flow, ctx)) => {
                self.kick_off(
                    "/class-plan",
                    format!("Starting /class-plan {}", pdf.display()),
                    flow,
                    ctx,
                );
            }
            Err(e) => self.log(format!("/class-plan failed to build flow: {e}")),
        }
    }

    pub fn start_student_edit(&mut self, name: String, edit_notes: String) {
        let date = Local::now().date_naive();
        let (flow, ctx) = student_edit_flow(self.root.clone(), date, name.clone(), edit_notes);
        self.kick_off(
            "/student-edit",
            format!("Starting /student-edit {name}…"),
            flow,
            ctx,
        );
    }

    fn kick_off(
        &mut self,
        flow_name: &str,
        log_line: String,
        flow: gt_flows::Flow,
        ctx: gt_flows::FlowCtx,
    ) {
        self.flow_name = Some(flow_name.into());
        self.flow_started_at = Some(std::time::Instant::now());
        self.steps.clear();
        self.selected_step = 0;
        self.log(log_line);
        let tools = Self::default_registry();
        let orch = Orchestrator::new(self.backend.clone(), tools);
        let handle = orch.start(flow, ctx);
        self.attach(handle);
        self.mode = AppMode::FlowActive;
    }

    fn attach(&mut self, h: OrchestratorHandle) {
        self.running_handle = Some(RunningFlow {
            flow_rx: h.flow_events,
            session_rxs: h.session_events,
            _join: h.join,
        });
    }
}

fn handle_flow_event(
    steps: &mut Vec<StepRow>,
    event: FlowEvent,
    flow_name: &mut Option<String>,
) {
    match event {
        FlowEvent::FlowStarted { name, steps: descs, .. } => {
            *flow_name = Some(name);
            *steps = descs.into_iter().map(step_row_from_desc).collect();
        }
        FlowEvent::StepStateChanged { step, state } => {
            if let Some(row) = steps.iter_mut().find(|s| s.id == step) {
                row.state = state;
            }
        }
        FlowEvent::StepArtifactProduced { step, key, path } => {
            if let Some(row) = steps.iter_mut().find(|s| s.id == step) {
                row.artifacts.push(format!("{key}: {path}"));
            }
        }
        FlowEvent::FlowDone { .. } => {}
    }
}

fn step_row_from_desc(d: StepDescriptor) -> StepRow {
    StepRow {
        id: d.id,
        name: d.name,
        kind: d.kind,
        state: StepState::Queued,
        streaming_text: String::new(),
        last_tool: None,
        artifacts: Vec::new(),
    }
}

fn handle_session_event(row: &mut StepRow, ev: SessionEvent) {
    match ev {
        SessionEvent::TokenDelta { text } => {
            row.streaming_text.push_str(&text);
            if row.streaming_text.len() > 8000 {
                let drop_n = row.streaming_text.len() - 8000;
                row.streaming_text.drain(..drop_n);
            }
        }
        SessionEvent::ToolCallStarted { tool, .. } => {
            row.last_tool = Some(tool);
        }
        SessionEvent::ToolCallResult { ok, output, .. } if !ok => {
            row.streaming_text.push_str("\n[tool error] ");
            row.streaming_text.push_str(&output);
        }
        SessionEvent::Failed { error, .. } => {
            row.state = StepState::Failed;
            row.streaming_text.push_str("\n[failed] ");
            row.streaming_text.push_str(&error);
        }
        SessionEvent::Done { .. } => {
            // Orchestrator will also send StepStateChanged(Done); leave state
            // changes to that signal so we don't race.
        }
        _ => {}
    }
}
