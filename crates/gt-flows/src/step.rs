use async_trait::async_trait;
use gt_core::ids::StepId;
use gt_core::session::SessionBuilder;
use std::path::PathBuf;
use thiserror::Error;

use crate::context::FlowCtx;

#[derive(Debug, Error)]
pub enum FlowError {
    #[error("step '{step}' failed: {msg}")]
    Step { step: String, msg: String },
    #[error("required artifact '{0}' missing")]
    MissingArtifact(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("internal: {0}")]
    Internal(String),
}

#[derive(Debug, Clone, Default)]
pub struct StepOutcome {
    /// Symbolic artifact keys → produced paths.
    pub outputs: Vec<(String, PathBuf)>,
}

#[async_trait]
pub trait DeterministicStep: Send + Sync {
    async fn run(&self, ctx: &FlowCtx) -> Result<StepOutcome, FlowError>;
}

pub trait AgentStepFactory: Send + Sync {
    /// Build the `SessionBuilder` for this step, given the flow context.
    /// The orchestrator wires the event sink and runs the session.
    fn build(&self, ctx: &FlowCtx) -> SessionBuilder;
    /// Declared output keys — the orchestrator records what the session
    /// produced (paths inside `working_dir`) under these names.
    fn output_keys(&self) -> Vec<(String, PathBuf)>;
}

pub enum StepKind {
    Deterministic(Box<dyn DeterministicStep>),
    Agent(Box<dyn AgentStepFactory>),
}

pub struct StepNode {
    pub id: StepId,
    pub name: String,
    pub kind: StepKind,
    pub parallel_group: Option<String>,
}

impl StepNode {
    pub fn det(name: impl Into<String>, s: impl DeterministicStep + 'static) -> Self {
        Self {
            id: StepId::new(),
            name: name.into(),
            kind: StepKind::Deterministic(Box::new(s)),
            parallel_group: None,
        }
    }
    pub fn agent(name: impl Into<String>, factory: impl AgentStepFactory + 'static) -> Self {
        Self {
            id: StepId::new(),
            name: name.into(),
            kind: StepKind::Agent(Box::new(factory)),
            parallel_group: None,
        }
    }
    pub fn in_group(mut self, group: impl Into<String>) -> Self {
        self.parallel_group = Some(group.into());
        self
    }
}

pub struct Flow {
    pub id: gt_core::ids::FlowId,
    pub name: String,
    pub steps: Vec<StepNode>,
}

impl Flow {
    pub fn new(name: impl Into<String>, steps: Vec<StepNode>) -> Self {
        Self {
            id: gt_core::ids::FlowId::new(),
            name: name.into(),
            steps,
        }
    }
}
