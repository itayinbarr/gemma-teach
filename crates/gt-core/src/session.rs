//! Session: one isolated agent run.
//!
//! Owns a turn loop that ties together backend → parser → quality monitor →
//! tool dispatch. Streams `SessionEvent`s to a caller-provided sink. No
//! terminal or filesystem I/O lives here directly — sessions are scoped to a
//! caller-provided `working_dir` that is forwarded to each tool's `execute`.

use futures::StreamExt;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::mpsc;

use crate::backend::{GenerateRequest, LlmBackend, TokenEvent};
use crate::ids::SessionId;
use crate::knowledge::{render_knowledge_block, KnowledgeInjector};
use crate::message::{ChatMessage, MessageRole, RawToolCall, ToolCallId};
use crate::model_profile::ModelProfile;
use crate::parser::{parse_assistant_output, ParsedToolCall};
use crate::quality::{CorrectionVerdict, QualityMonitor, RecentCalls};
use crate::session_event::{
    QualityIssueKind, SessionErrorKind, SessionEvent, SessionOutcomeSummary, SteerReason,
};
use crate::skills::{render_skills_block, RecentUsage, SelectionCtx, SkillInjector};
use crate::tool::{ToolCtx, ToolRegistry};

/// Event-stream sink. A cheap wrapper around `mpsc::Sender` so callers without
/// an event consumer can pass `EventSink::noop()`.
#[derive(Clone)]
pub struct EventSink {
    inner: Option<mpsc::Sender<SessionEvent>>,
}

impl EventSink {
    pub fn channel(buffer: usize) -> (Self, mpsc::Receiver<SessionEvent>) {
        let (tx, rx) = mpsc::channel(buffer);
        (Self { inner: Some(tx) }, rx)
    }
    pub fn noop() -> Self {
        Self { inner: None }
    }
    pub async fn emit(&self, event: SessionEvent) {
        if let Some(tx) = &self.inner {
            let _ = tx.send(event).await;
        }
    }
}

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("backend error: {0}")]
    Backend(String),
    #[error("turn cap exceeded ({0})")]
    TurnCap(u32),
    #[error("correction loop — model failed quality checks repeatedly")]
    CorrectionLoop,
    #[error("session aborted")]
    Aborted,
    #[error("internal: {0}")]
    Internal(String),
}

impl SessionError {
    fn kind(&self) -> SessionErrorKind {
        match self {
            SessionError::Backend(_) => SessionErrorKind::BackendError,
            SessionError::TurnCap(_) => SessionErrorKind::TurnCapExceeded,
            SessionError::CorrectionLoop => SessionErrorKind::CorrectionLoop,
            SessionError::Aborted => SessionErrorKind::Aborted,
            SessionError::Internal(_) => SessionErrorKind::Internal,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SessionOutcome {
    pub id: SessionId,
    pub turns: u32,
    pub tool_calls: u32,
    pub final_message: Option<String>,
}

pub struct SessionBuilder {
    pub id: SessionId,
    pub name: String,
    pub working_dir: PathBuf,
    pub system_prompt: String,
    pub task_prompt: String,
    pub allowed_tools: Vec<String>,
    pub model_profile: ModelProfile,
    pub event_sink: EventSink,
    pub allow_bare_json_in_parser: bool,
    /// Optional skill cards (loaded by the caller and filtered to allowed tools).
    pub skill_injector: Option<Arc<SkillInjector>>,
    pub knowledge_injector: Option<Arc<KnowledgeInjector>>,
}

impl SessionBuilder {
    pub fn new(name: impl Into<String>, working_dir: PathBuf) -> Self {
        Self {
            id: SessionId::new(),
            name: name.into(),
            working_dir,
            system_prompt: String::new(),
            task_prompt: String::new(),
            allowed_tools: Vec::new(),
            model_profile: ModelProfile::default(),
            event_sink: EventSink::noop(),
            allow_bare_json_in_parser: true,
            skill_injector: None,
            knowledge_injector: None,
        }
    }
    pub fn system_prompt(mut self, p: impl Into<String>) -> Self {
        self.system_prompt = p.into();
        self
    }
    pub fn task_prompt(mut self, p: impl Into<String>) -> Self {
        self.task_prompt = p.into();
        self
    }
    pub fn allowed_tools<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_tools = names.into_iter().map(Into::into).collect();
        self
    }
    pub fn model_profile(mut self, p: ModelProfile) -> Self {
        self.model_profile = p;
        self
    }
    pub fn event_sink(mut self, sink: EventSink) -> Self {
        self.event_sink = sink;
        self
    }
    pub fn skill_injector(mut self, inj: Arc<SkillInjector>) -> Self {
        self.skill_injector = Some(inj);
        self
    }
    pub fn knowledge_injector(mut self, inj: Arc<KnowledgeInjector>) -> Self {
        self.knowledge_injector = Some(inj);
        self
    }

    pub async fn run(
        self,
        backend: Arc<dyn LlmBackend>,
        tool_registry: ToolRegistry,
    ) -> Result<SessionOutcome, SessionError> {
        let allowed_registry = tool_registry.allowed_subset(&self.allowed_tools);
        let known: HashSet<String> = allowed_registry.names().into_iter().collect();

        backend
            .load()
            .await
            .map_err(|e| SessionError::Backend(e.to_string()))?;

        self.event_sink
            .emit(SessionEvent::SessionStarted {
                id: self.id,
                name: self.name.clone(),
            })
            .await;

        let mut quality = QualityMonitor::new(self.model_profile.max_consecutive_corrections);
        let mut recent_usage = RecentUsage::new(4);
        let mut recent_calls = RecentCalls::default();
        let mut last_failed_tool: Option<String> = None;
        let mut pending_steers: Vec<(SteerReason, String)> = Vec::new();
        let mut tool_calls_total: u32 = 0;
        let mut final_message: Option<String> = None;

        // The history starts with the task as a user message. The system
        // prompt is rebuilt per-turn and prepended to that conversation as
        // the first message in the request payload.
        let mut history: Vec<ChatMessage> = vec![ChatMessage::user(&self.task_prompt)];

        for turn_index in 0..self.model_profile.turn_cap {
            let turn = turn_index + 1;
            self.event_sink.emit(SessionEvent::TurnStarted { turn }).await;

            // ----- Build the system prompt for this turn -----
            let mut system_prompt = self.system_prompt.clone();
            if let Some(inj) = &self.knowledge_injector {
                let cards =
                    inj.select(&self.task_prompt, self.model_profile.knowledge_token_budget);
                let block = render_knowledge_block(&cards);
                if !block.is_empty() {
                    system_prompt.push_str("\n\n");
                    system_prompt.push_str(&block);
                }
            }
            if let Some(inj) = &self.skill_injector {
                let ctx = SelectionCtx {
                    user_prompt: &self.task_prompt,
                    last_failed_tool: last_failed_tool.as_deref(),
                    recent: &recent_usage,
                    token_budget: self.model_profile.skill_token_budget,
                    allowed_tools: &self.allowed_tools,
                };
                let cards = inj.select(&ctx);
                let block = render_skills_block(&cards);
                if !block.is_empty() {
                    system_prompt.push_str("\n\n");
                    system_prompt.push_str(&block);
                }
            }
            // Apply steers from the previous iteration. We deliver them as a
            // single `<system>`-style trailing block on the system prompt so
            // the model sees the correction *before* it generates this turn
            // (this is the "steer" delivery mode described in little-coder).
            if !pending_steers.is_empty() {
                system_prompt.push_str("\n\n## Correction\n");
                for (_reason, msg) in &pending_steers {
                    system_prompt.push_str(msg);
                    system_prompt.push('\n');
                }
                pending_steers.clear();
            }

            let mut messages = Vec::with_capacity(history.len() + 1);
            messages.push(ChatMessage::system(&system_prompt));
            messages.extend(history.iter().cloned());

            let req = GenerateRequest {
                messages,
                tools: allowed_registry.schemas(),
                max_tokens: self.model_profile.max_tokens,
                temperature: self.model_profile.temperature,
                stop: Vec::new(),
                thinking_budget: Some(self.model_profile.thinking_budget),
            };

            // ----- Stream from backend -----
            let mut stream = backend
                .generate(req)
                .await
                .map_err(|e| SessionError::Backend(e.to_string()))?;

            let mut text_buf = String::new();
            let mut native_calls: Vec<RawToolCall> = Vec::new();
            while let Some(ev) = stream.next().await {
                match ev {
                    TokenEvent::Delta { text } => {
                        text_buf.push_str(&text);
                        self.event_sink
                            .emit(SessionEvent::TokenDelta { text })
                            .await;
                    }
                    TokenEvent::ThinkingDelta { text } => {
                        self.event_sink
                            .emit(SessionEvent::ThinkingDelta { text })
                            .await;
                    }
                    TokenEvent::ToolCall { call } => {
                        native_calls.push(call);
                    }
                    TokenEvent::Done { .. } => break,
                }
            }

            // ----- Parse embedded calls -----
            let parsed = parse_assistant_output(&text_buf, self.allow_bare_json_in_parser);
            // Merge native + parsed; native go first to preserve backend-side order.
            let mut all_calls: Vec<ParsedToolCall> = Vec::new();
            for nc in &native_calls {
                all_calls.push(ParsedToolCall {
                    call: nc.clone(),
                    source: crate::parser::CallSource::Native,
                    had_repair: false,
                    unrepairable: false,
                });
            }
            all_calls.extend(parsed.tool_calls.clone());

            let outcome_for_quality = crate::parser::ParseOutcome {
                text: parsed.text.clone(),
                thinking: parsed.thinking.clone(),
                tool_calls: all_calls.clone(),
                steer_reasons: parsed.steer_reasons.clone(),
            };

            // Surface any embedded-call steer reasons as events + queue a nudge.
            for reason in &parsed.steer_reasons {
                self.event_sink
                    .emit(SessionEvent::Steer {
                        reason: *reason,
                        message: embedded_call_nudge(),
                    })
                    .await;
                pending_steers.push((*reason, embedded_call_nudge()));
            }

            // ----- Quality monitor -----
            let (issues, verdict) = quality.inspect(&outcome_for_quality, &known, &recent_calls);
            for issue in &issues {
                self.event_sink
                    .emit(SessionEvent::QualityIssue {
                        issue: issue.kind.clone(),
                        action: issue.action.clone(),
                    })
                    .await;
                pending_steers.push((SteerReason::QualityCorrection, issue.action.message.clone()));
            }
            if matches!(verdict, CorrectionVerdict::Aborting) {
                let err = SessionError::CorrectionLoop;
                self.event_sink
                    .emit(SessionEvent::Failed {
                        error: err.to_string(),
                        error_kind: err.kind(),
                    })
                    .await;
                return Err(err);
            }

            // If issues exist but verdict is Correctable, do not append this
            // assistant turn to history at all — it would just preserve the
            // bad output. Loop with the steers applied.
            if !issues.is_empty() {
                continue;
            }

            // ----- Record the assistant turn -----
            // Even if there were embedded-call steers, the calls themselves
            // are valid; we still execute them, and the model is nudged for next turn.
            let assistant_calls: Vec<RawToolCall> = all_calls.iter().map(|p| p.call.clone()).collect();
            history.push(ChatMessage::assistant_with_calls(
                parsed.text.clone(),
                assistant_calls.clone(),
            ));

            // ----- If no tool calls, this is the end of the session -----
            if assistant_calls.is_empty() {
                final_message = if parsed.text.is_empty() {
                    None
                } else {
                    Some(parsed.text.clone())
                };
                break;
            }

            // ----- Execute each tool call -----
            let mut any_tool_error = false;
            let mut next_recent: Vec<(String, serde_json::Value)> = Vec::new();
            for call in &assistant_calls {
                let tool = match allowed_registry.get(&call.name) {
                    Some(t) => t,
                    None => {
                        // Quality monitor should have caught this; if it didn't,
                        // surface as a tool error and continue.
                        let msg = format!("Tool '{}' is not allowed in this session.", call.name);
                        self.event_sink
                            .emit(SessionEvent::ToolCallResult {
                                call_id: call.id.0.clone(),
                                ok: false,
                                output: msg.clone(),
                                truncated: false,
                            })
                            .await;
                        history.push(ChatMessage::tool_result(
                            ToolCallId(call.id.0.clone()),
                            &call.name,
                            msg,
                        ));
                        any_tool_error = true;
                        last_failed_tool = Some(call.name.clone());
                        continue;
                    }
                };
                self.event_sink
                    .emit(SessionEvent::ToolCallStarted {
                        call_id: call.id.0.clone(),
                        tool: call.name.clone(),
                        args_json: call.args.clone(),
                    })
                    .await;

                let ctx = ToolCtx {
                    working_dir: &self.working_dir,
                    session_id: self.id,
                };
                let result = tool.execute(call.args.clone(), &ctx).await;

                tool_calls_total += 1;
                recent_usage.record(call.name.clone());
                next_recent.push((call.name.clone(), call.args.clone()));

                if result.is_error {
                    any_tool_error = true;
                    last_failed_tool = Some(call.name.clone());
                } else {
                    // success — clear the last-failed gate for that tool if it matched
                    if last_failed_tool.as_deref() == Some(call.name.as_str()) {
                        last_failed_tool = None;
                    }
                }

                self.event_sink
                    .emit(SessionEvent::ToolCallResult {
                        call_id: call.id.0.clone(),
                        ok: !result.is_error,
                        output: result.output.clone(),
                        truncated: false,
                    })
                    .await;

                let mut tool_msg = result.output;
                if let Some(s) = result.suggestion {
                    tool_msg.push_str("\n\n");
                    tool_msg.push_str(&s);
                }
                history.push(ChatMessage::tool_result(
                    ToolCallId(call.id.0.clone()),
                    &call.name,
                    tool_msg,
                ));
            }

            recent_calls.previous = next_recent;
            // suppress unused warning when no tool errors
            let _ = any_tool_error;
        }

        let outcome = SessionOutcome {
            id: self.id,
            turns: 0, // filled below
            tool_calls: tool_calls_total,
            final_message: final_message.clone(),
        };

        // Decide between Done and TurnCap based on whether we exited the loop
        // via the no-tool-calls branch (in which case `final_message` may be
        // set OR the model just said nothing).
        // We treat reaching the end of the loop with no break as turn-cap.
        // Track via a sentinel: if assistant final break happened, final_message
        // is `Some(_)` OR history's last message is from the assistant with no tool calls.
        let last_was_done = history
            .last()
            .map(|m| m.role == MessageRole::Assistant && m.tool_calls.is_empty())
            .unwrap_or(false);

        if !last_was_done {
            let err = SessionError::TurnCap(self.model_profile.turn_cap);
            self.event_sink
                .emit(SessionEvent::Failed {
                    error: err.to_string(),
                    error_kind: err.kind(),
                })
                .await;
            return Err(err);
        }

        let turns_used = history
            .iter()
            .filter(|m| m.role == MessageRole::Assistant)
            .count() as u32;

        self.event_sink
            .emit(SessionEvent::Done {
                outcome: SessionOutcomeSummary {
                    turns: turns_used,
                    tool_calls: tool_calls_total,
                    success: true,
                    final_message: final_message.clone(),
                },
            })
            .await;

        Ok(SessionOutcome {
            turns: turns_used,
            ..outcome
        })
    }
}

fn embedded_call_nudge() -> String {
    "Your tool call was embedded in text (code fence or XML tag). Re-issue tool calls using the native tool-call channel — emit them as structured calls, not inside ``` blocks or <tool_call> tags.".into()
}

// quiet the warning for the unused QualityIssueKind enum import path when
// nothing in this file refers to it directly.
const _: fn() = || {
    let _ = QualityIssueKind::EmptyResponse;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{MockBackend, MockScript, StopReason};
    use crate::tool::{ParameterKind, ParameterSchema, Tool, ToolCtx, ToolResult, ToolSchema};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct RecordingWrite {
        calls: AtomicU32,
    }
    #[async_trait]
    impl Tool for RecordingWrite {
        fn name(&self) -> &'static str {
            "Write"
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "Write".into(),
                description: "write file".into(),
                parameters: vec![
                    ParameterSchema {
                        name: "path".into(),
                        kind: ParameterKind::String,
                        required: true,
                        description: "path".into(),
                    },
                    ParameterSchema {
                        name: "content".into(),
                        kind: ParameterKind::String,
                        required: true,
                        description: "content".into(),
                    },
                ],
            }
        }
        async fn execute(&self, args: serde_json::Value, _ctx: &ToolCtx<'_>) -> ToolResult {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let path = args
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            ToolResult::ok(format!("wrote {}", path))
        }
    }

    fn build_registry() -> (ToolRegistry, Arc<RecordingWrite>) {
        let t = Arc::new(RecordingWrite {
            calls: AtomicU32::new(0),
        });
        let r = ToolRegistry::new().register(t.clone());
        (r, t)
    }

    #[tokio::test]
    async fn session_runs_a_native_tool_call_then_done() {
        let backend = Arc::new(MockBackend::new());
        backend.push(
            MockScript::new()
                .tool(
                    "Write",
                    serde_json::json!({"path":"out.md","content":"hi"}),
                )
                .done(StopReason::Eos),
        );
        backend.push(MockScript::new().text("Done.").done(StopReason::Eos));

        let (reg, write_tool) = build_registry();

        let outcome = SessionBuilder::new("test", std::env::temp_dir())
            .allowed_tools(["Write"])
            .system_prompt("test session")
            .task_prompt("write out.md")
            .model_profile(ModelProfile::test_default())
            .run(backend, reg)
            .await
            .unwrap();

        assert_eq!(write_tool.calls.load(Ordering::SeqCst), 1);
        assert_eq!(outcome.tool_calls, 1);
        assert_eq!(outcome.final_message.as_deref(), Some("Done."));
    }

    #[tokio::test]
    async fn session_parses_fenced_tool_call_and_executes_it() {
        let backend = Arc::new(MockBackend::new());
        backend.push(
            MockScript::new()
                .text("I'll save the file.\n```tool\n{\"name\":\"Write\",\"args\":{\"path\":\"a.md\",\"content\":\"x\"}}\n```")
                .done(StopReason::Eos),
        );
        backend.push(MockScript::new().text("Done.").done(StopReason::Eos));

        let (reg, write_tool) = build_registry();
        let outcome = SessionBuilder::new("t", std::env::temp_dir())
            .allowed_tools(["Write"])
            .task_prompt("write a.md")
            .model_profile(ModelProfile::test_default())
            .run(backend, reg)
            .await
            .unwrap();
        assert_eq!(write_tool.calls.load(Ordering::SeqCst), 1);
        assert_eq!(outcome.tool_calls, 1);
    }

    #[tokio::test]
    async fn hallucinated_tool_triggers_correction() {
        let backend = Arc::new(MockBackend::new());
        backend.push(MockScript::new().tool("DoesNotExist", serde_json::json!({})).done(StopReason::Eos));
        backend.push(
            MockScript::new()
                .tool("Write", serde_json::json!({"path":"a.md","content":"x"}))
                .done(StopReason::Eos),
        );
        backend.push(MockScript::new().text("Done.").done(StopReason::Eos));

        let (reg, write_tool) = build_registry();
        let outcome = SessionBuilder::new("t", std::env::temp_dir())
            .allowed_tools(["Write"])
            .task_prompt("hello")
            .model_profile(ModelProfile::test_default())
            .run(backend, reg)
            .await
            .unwrap();
        // The first turn was a hallucination → corrected → model retried correctly.
        assert_eq!(write_tool.calls.load(Ordering::SeqCst), 1);
        assert!(outcome.turns >= 2);
    }

    #[tokio::test]
    async fn repeated_failure_aborts_with_correction_loop() {
        let backend = Arc::new(MockBackend::new());
        // 3 empty responses in a row → quality monitor will abort after 3rd.
        backend.push(MockScript::new().done(StopReason::Eos));
        backend.push(MockScript::new().done(StopReason::Eos));
        backend.push(MockScript::new().done(StopReason::Eos));

        let (reg, _) = build_registry();
        let err = SessionBuilder::new("t", std::env::temp_dir())
            .allowed_tools(["Write"])
            .task_prompt("hello")
            .model_profile(ModelProfile::test_default())
            .run(backend, reg)
            .await
            .unwrap_err();
        assert!(matches!(err, SessionError::CorrectionLoop));
    }
}
