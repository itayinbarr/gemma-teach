//! Inference backend abstraction.
//!
//! `LlmBackend` is the seam between the harness and the model runtime.
//! Three impls live in this crate: a deterministic `MockBackend` for tests,
//! a chatty `EchoBackend` for TUI smoke runs, and (behind `backend-llama`)
//! a real `LlamaCppBackend` (wired in C10).

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use thiserror::Error;

use crate::message::{ChatMessage, RawToolCall, ToolCallId};
use crate::tool::ToolSchema;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolSchema>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub stop: Vec<String>,
    pub thinking_budget: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Eos,
    MaxTokens,
    Stop,
    ThinkingBudgetExceeded,
    BackendAborted,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub thinking_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TokenEvent {
    Delta { text: String },
    ThinkingDelta { text: String },
    /// Native tool-call channel (backends that support it). Backends that don't
    /// only emit text Deltas and rely on the parser to extract calls.
    ToolCall { call: RawToolCall },
    Done { stop_reason: StopReason, usage: Usage },
}

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("backend not loaded")]
    NotLoaded,
    #[error("model not found at path: {0}")]
    ModelNotFound(String),
    #[error("backend i/o error: {0}")]
    Io(String),
    #[error("backend failed to generate: {0}")]
    Generation(String),
}

#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// Idempotent — safe to call repeatedly.
    async fn load(&self) -> Result<(), BackendError>;
    fn is_loaded(&self) -> bool;
    async fn generate(
        &self,
        req: GenerateRequest,
    ) -> Result<BoxStream<'static, TokenEvent>, BackendError>;
}

// -----------------------------------------------------------------------------
// MockBackend: deterministic scripted backend for tests.
//
// Scenarios are keyed by a hash of the latest user message OR by an
// explicit name passed at construction. Each scenario is a sequence of
// `TokenEvent`s to replay verbatim.
// -----------------------------------------------------------------------------

/// One scripted scenario. The backend emits these events in order, terminated
/// by an implicit `Done` if the last event is not already `Done`.
#[derive(Debug, Clone, Default)]
pub struct MockScript {
    pub events: Vec<TokenEvent>,
}

impl MockScript {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn text(mut self, t: impl Into<String>) -> Self {
        self.events.push(TokenEvent::Delta { text: t.into() });
        self
    }

    pub fn tool(mut self, name: &str, args: serde_json::Value) -> Self {
        self.events.push(TokenEvent::ToolCall {
            call: RawToolCall {
                id: ToolCallId::new(),
                name: name.into(),
                args,
            },
        });
        self
    }

    pub fn thinking(mut self, t: impl Into<String>) -> Self {
        self.events.push(TokenEvent::ThinkingDelta { text: t.into() });
        self
    }

    pub fn done(mut self, reason: StopReason) -> Self {
        self.events.push(TokenEvent::Done {
            stop_reason: reason,
            usage: Usage::default(),
        });
        self
    }
}

#[derive(Default)]
pub struct MockBackend {
    /// Sequence of scripts replayed FIFO across successive calls.
    /// Indexed by `next_index`.
    scripts: Mutex<Vec<MockScript>>,
    /// Optional scenario selector keyed on (concatenated) user message hashes.
    keyed: Mutex<HashMap<String, MockScript>>,
    next_index: Mutex<usize>,
    loaded: Mutex<bool>,
}

impl MockBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a script that will be returned on the next `generate` call.
    pub fn push(&self, script: MockScript) {
        self.scripts.lock().unwrap().push(script);
    }

    /// Register a script keyed by an exact-match user message substring.
    pub fn on_user(&self, needle: impl Into<String>, script: MockScript) {
        self.keyed.lock().unwrap().insert(needle.into(), script);
    }

    fn pick_script(&self, req: &GenerateRequest) -> MockScript {
        // 1) keyed lookup
        let latest_user = req
            .messages
            .iter()
            .rev()
            .find(|m| matches!(m.role, crate::message::MessageRole::User))
            .map(|m| m.content.clone())
            .unwrap_or_default();
        let keyed = self.keyed.lock().unwrap();
        for (needle, script) in keyed.iter() {
            if latest_user.contains(needle) {
                return script.clone();
            }
        }
        drop(keyed);
        // 2) FIFO from queue
        let mut idx = self.next_index.lock().unwrap();
        let scripts = self.scripts.lock().unwrap();
        if let Some(s) = scripts.get(*idx) {
            *idx += 1;
            return s.clone();
        }
        // 3) default: empty assistant message → quality monitor will catch this
        MockScript::new().done(StopReason::Eos)
    }
}

#[async_trait]
impl LlmBackend for MockBackend {
    async fn load(&self) -> Result<(), BackendError> {
        *self.loaded.lock().unwrap() = true;
        Ok(())
    }
    fn is_loaded(&self) -> bool {
        *self.loaded.lock().unwrap()
    }
    async fn generate(
        &self,
        req: GenerateRequest,
    ) -> Result<BoxStream<'static, TokenEvent>, BackendError> {
        let mut script = self.pick_script(&req);
        let has_done = script
            .events
            .iter()
            .any(|e| matches!(e, TokenEvent::Done { .. }));
        if !has_done {
            script.events.push(TokenEvent::Done {
                stop_reason: StopReason::Eos,
                usage: Usage::default(),
            });
        }
        let stream = futures::stream::iter(script.events.into_iter()).boxed();
        Ok(stream)
    }
}

// -----------------------------------------------------------------------------
// EchoBackend: emits the latest user message back as a single delta. Used for
// smoke-running the TUI without a real model.
// -----------------------------------------------------------------------------

#[derive(Default)]
pub struct EchoBackend;

impl EchoBackend {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl LlmBackend for EchoBackend {
    async fn load(&self) -> Result<(), BackendError> {
        Ok(())
    }
    fn is_loaded(&self) -> bool {
        true
    }
    async fn generate(
        &self,
        req: GenerateRequest,
    ) -> Result<BoxStream<'static, TokenEvent>, BackendError> {
        let last_user = req
            .messages
            .iter()
            .rev()
            .find(|m| matches!(m.role, crate::message::MessageRole::User))
            .map(|m| m.content.clone())
            .unwrap_or_else(|| "(empty)".into());
        let stream = futures::stream::iter(vec![
            TokenEvent::Delta {
                text: format!("echo: {}", last_user),
            },
            TokenEvent::Done {
                stop_reason: StopReason::Eos,
                usage: Usage::default(),
            },
        ])
        .boxed();
        Ok(stream)
    }
}

/// Convenience: wrap a backend in an Arc.
pub fn shared<B: LlmBackend + 'static>(b: B) -> Arc<dyn LlmBackend> {
    Arc::new(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ChatMessage;

    fn req(msgs: Vec<ChatMessage>) -> GenerateRequest {
        GenerateRequest {
            messages: msgs,
            tools: vec![],
            max_tokens: 64,
            temperature: 0.0,
            stop: vec![],
            thinking_budget: None,
        }
    }

    #[tokio::test]
    async fn mock_backend_replays_scripts_in_order() {
        let b = MockBackend::new();
        b.push(MockScript::new().text("first").done(StopReason::Eos));
        b.push(MockScript::new().text("second").done(StopReason::Eos));
        b.load().await.unwrap();

        let mut s1 = b.generate(req(vec![ChatMessage::user("hi")])).await.unwrap();
        let mut texts = vec![];
        while let Some(e) = s1.next().await {
            if let TokenEvent::Delta { text } = e {
                texts.push(text);
            }
        }
        assert_eq!(texts, vec!["first"]);

        let mut s2 = b.generate(req(vec![ChatMessage::user("hi")])).await.unwrap();
        let mut texts = vec![];
        while let Some(e) = s2.next().await {
            if let TokenEvent::Delta { text } = e {
                texts.push(text);
            }
        }
        assert_eq!(texts, vec!["second"]);
    }

    #[tokio::test]
    async fn mock_backend_keyed_match_takes_priority() {
        let b = MockBackend::new();
        b.push(MockScript::new().text("default"));
        b.on_user("special", MockScript::new().text("keyed"));
        b.load().await.unwrap();

        let mut s = b
            .generate(req(vec![ChatMessage::user("please do something special now")]))
            .await
            .unwrap();
        let mut texts = vec![];
        while let Some(e) = s.next().await {
            if let TokenEvent::Delta { text } = e {
                texts.push(text);
            }
        }
        assert_eq!(texts, vec!["keyed"]);
    }

    #[tokio::test]
    async fn echo_backend_returns_last_user() {
        let b = EchoBackend::new();
        let mut s = b.generate(req(vec![ChatMessage::user("ping")])).await.unwrap();
        let mut texts = vec![];
        while let Some(e) = s.next().await {
            if let TokenEvent::Delta { text } = e {
                texts.push(text);
            }
        }
        assert_eq!(texts, vec!["echo: ping"]);
    }
}
