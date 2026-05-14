use serde::{Deserialize, Serialize};

use crate::ids::{FlowId, SessionId, StepId};

/// Streamed wire format from a running session to its frontend.
/// Stable across releases; new variants must be additive.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionEvent {
    SessionStarted {
        id: SessionId,
        name: String,
    },
    TurnStarted {
        turn: u32,
    },
    TokenDelta {
        text: String,
    },
    ThinkingDelta {
        text: String,
    },
    ToolCallStarted {
        call_id: String,
        tool: String,
        args_json: serde_json::Value,
    },
    ToolCallResult {
        call_id: String,
        ok: bool,
        output: String,
        truncated: bool,
    },
    QualityIssue {
        issue: QualityIssueKind,
        action: CorrectionAction,
    },
    Steer {
        reason: SteerReason,
        message: String,
    },
    Done {
        outcome: SessionOutcomeSummary,
    },
    Failed {
        error: String,
        error_kind: SessionErrorKind,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "issue", rename_all = "snake_case")]
pub enum QualityIssueKind {
    EmptyResponse,
    EmptyToolName,
    HallucinatedTool { name: String },
    RepeatedToolCall { tool: String },
    MalformedArgs { tool: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionAction {
    pub message: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SteerReason {
    EmbeddedToolCall,
    QualityCorrection,
    ThinkingBudgetExceeded,
    UserSteer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionOutcomeSummary {
    pub turns: u32,
    pub tool_calls: u32,
    pub success: bool,
    pub final_message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionErrorKind {
    BackendError,
    ToolError,
    CorrectionLoop,
    TurnCapExceeded,
    ThinkingBudgetExceeded,
    Aborted,
    Internal,
}

/// Streamed wire format from a running flow to its frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FlowEvent {
    FlowStarted {
        id: FlowId,
        name: String,
        steps: Vec<StepDescriptor>,
    },
    StepStateChanged {
        step: StepId,
        state: StepState,
    },
    StepArtifactProduced {
        step: StepId,
        key: String,
        path: String,
    },
    FlowDone {
        id: FlowId,
        ok: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepDescriptor {
    pub id: StepId,
    pub name: String,
    pub kind: String, // "deterministic" | "agent"
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepState {
    Queued,
    Running,
    Streaming,
    Done,
    Failed,
}
