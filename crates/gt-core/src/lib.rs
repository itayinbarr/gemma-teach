//! Gemma Teach engine core.
//!
//! Model-agnostic agent loop: inference abstraction, deterministic output parser,
//! quality monitor, per-turn skill/knowledge injection, tool registry, session
//! orchestration. No filesystem-rooted paths, no terminal I/O — those live in
//! the binary crates so this library can be linked into the iOS app via uniffi.

pub mod backend;
pub mod ids;
pub mod message;
pub mod model_profile;
pub mod knowledge;
#[cfg(feature = "backend-llama")]
pub mod llama_backend;
#[cfg(feature = "model-fetch")]
pub mod model_fetch;
pub mod parser;
pub mod quality;
pub mod session;
pub mod skills;
pub mod tool;
pub mod session_event;

pub use session::{EventSink, SessionBuilder, SessionError, SessionOutcome};

pub use knowledge::{parse_knowledge_markdown, render_knowledge_block, KnowledgeCard, KnowledgeInjector};
pub use parser::{parse_assistant_output, repair_and_parse, CallSource, ParseOutcome, ParsedToolCall};
pub use quality::{CorrectionVerdict, QualityIssue, QualityMonitor, RecentCalls};
pub use skills::{
    parse_skill_markdown, render_skills_block, RecentUsage, SelectionCtx, SkillCard, SkillInjector,
};

pub use backend::{
    BackendError, EchoBackend, GenerateRequest, LlmBackend, MockBackend, MockScript,
    StopReason, TokenEvent, Usage,
};
pub use ids::{FlowId, SessionId, StepId};
pub use message::{ChatMessage, MessageRole, RawToolCall, ToolCallId};
pub use model_profile::ModelProfile;
pub use session_event::{
    CorrectionAction, FlowEvent, QualityIssueKind, SessionErrorKind, SessionEvent,
    SessionOutcomeSummary, StepState, SteerReason,
};
pub use tool::{
    ParameterKind, ParameterSchema, Tool, ToolCtx, ToolRegistry, ToolResult, ToolSchema,
};
