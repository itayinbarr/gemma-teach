//! Pre-defined flow pipelines for Gemma Teach features.
//!
//! Each `Flow` is a DAG of `StepNode`s. Steps are either deterministic
//! (pure Rust, no model) or agent sessions (isolated harness runs with
//! a scoped working directory and narrow tool registry). Sessions are spawned
//! one-shot per step and emit their own `SessionEvent` stream on a child
//! channel.

pub mod artifacts;
pub mod context;
pub mod orchestrator;
pub mod step;

pub mod class_plan;
pub mod student_add;
pub mod student_edit;

pub use class_plan::ClassPlanSource;

pub use artifacts::{ArtifactKey, ArtifactMap};
pub use context::FlowCtx;
pub use orchestrator::{Orchestrator, OrchestratorHandle};
pub use step::{
    AgentStepFactory, DeterministicStep, Flow, FlowError, StepKind, StepNode, StepOutcome,
};
