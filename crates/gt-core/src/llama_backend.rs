//! Real llama.cpp-backed inference. Behind the `backend-llama` feature.
//!
//! Loads a GGUF model lazily via `llama-cpp-2` with Metal acceleration on Apple
//! Silicon. Streams tokens through the standard `LlmBackend::generate` contract.
//!
//! Tool-call detection: Gemma 4 does not expose first-class tool-call tokens,
//! so this backend only emits text deltas. The session's parser is responsible
//! for extracting tool calls from the text stream (fenced blocks, XML tags, or
//! bare JSON — see `parser.rs`).

#![cfg(feature = "backend-llama")]
#![allow(deprecated)]

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{Mutex, OnceCell};

/// Install the llama.cpp → tracing log bridge exactly once. Without this,
/// llama.cpp writes loader / Metal / context messages directly to stderr,
/// which (under a ratatui alternate-screen TUI) corrupts the rendered
/// layout. With it, the same messages flow through `tracing` and land in
/// whatever sink the binary configured (a log file, in our case).
static LOG_BRIDGE: std::sync::Once = std::sync::Once::new();
fn ensure_log_bridge() {
    LOG_BRIDGE.call_once(|| {
        llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
    });
}

use crate::backend::{
    BackendError, GenerateRequest, LlmBackend, StopReason, TokenEvent, Usage,
};
use crate::message::MessageRole;

/// Configuration for the llama.cpp backend.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub model_path: PathBuf,
    pub n_ctx: u32,
    pub n_gpu_layers: u32,
}

impl LlamaConfig {
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            // 32K context — Gemma 4 E2B is small enough that KV cache at this
            // size is comfortable on M-series, and /class-plan tailoring
            // sessions easily pre-load 4K+ tokens of student + lesson context.
            n_ctx: 32_768,
            n_gpu_layers: 999, // offload everything to Metal
        }
    }
}

struct Loaded {
    backend: LlamaBackend,
    model: LlamaModel,
}

pub struct LlamaCppBackend {
    cfg: LlamaConfig,
    loaded: OnceCell<Arc<Loaded>>,
    /// Inference serialization. llama.cpp's context isn't safe to call
    /// concurrently for a single model, so we serialize generates.
    lock: Mutex<()>,
}

impl LlamaCppBackend {
    pub fn new(cfg: LlamaConfig) -> Self {
        Self {
            cfg,
            loaded: OnceCell::new(),
            lock: Mutex::new(()),
        }
    }

    async fn ensure_loaded(&self) -> Result<Arc<Loaded>, BackendError> {
        if let Some(l) = self.loaded.get() {
            return Ok(l.clone());
        }
        let cfg = self.cfg.clone();
        let loaded = tokio::task::spawn_blocking(move || {
            if !cfg.model_path.exists() {
                return Err(BackendError::ModelNotFound(cfg.model_path.display().to_string()));
            }
            ensure_log_bridge();
            let backend = LlamaBackend::init().map_err(|e| BackendError::Io(e.to_string()))?;
            let model_params = LlamaModelParams::default().with_n_gpu_layers(cfg.n_gpu_layers);
            let model = LlamaModel::load_from_file(&backend, &cfg.model_path, &model_params)
                .map_err(|e| BackendError::Io(format!("load model: {e}")))?;
            Ok(Arc::new(Loaded { backend, model }))
        })
        .await
        .map_err(|e| BackendError::Io(format!("join: {e}")))??;

        let _ = self.loaded.set(loaded.clone());
        Ok(loaded)
    }
}

#[async_trait]
impl LlmBackend for LlamaCppBackend {
    async fn load(&self) -> Result<(), BackendError> {
        self.ensure_loaded().await.map(|_| ())
    }

    fn is_loaded(&self) -> bool {
        self.loaded.get().is_some()
    }

    async fn generate(
        &self,
        req: GenerateRequest,
    ) -> Result<BoxStream<'static, TokenEvent>, BackendError> {
        let loaded = self.ensure_loaded().await?;
        let _guard = self.lock.lock().await;

        let prompt = render_chat_prompt(&req);
        let n_ctx = self.cfg.n_ctx;
        let max_tokens = req.max_tokens.max(1);
        let temperature = req.temperature.max(0.0);

        // Channel to stream tokens out of the blocking inference thread.
        let (tx, rx) = tokio::sync::mpsc::channel::<TokenEvent>(64);

        let loaded_cloned = loaded.clone();
        tokio::task::spawn_blocking(move || {
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(n_ctx));

            let model = &loaded_cloned.model;
            let mut ctx: LlamaContext = match model.new_context(&loaded_cloned.backend, ctx_params)
            {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.blocking_send(TokenEvent::Done {
                        stop_reason: StopReason::BackendAborted,
                        usage: Usage::default(),
                    });
                    let _ = e;
                    return;
                }
            };

            let tokens_list = match model.str_to_token(&prompt, AddBos::Always) {
                Ok(v) => v,
                Err(_) => {
                    let _ = tx.blocking_send(TokenEvent::Done {
                        stop_reason: StopReason::BackendAborted,
                        usage: Usage::default(),
                    });
                    return;
                }
            };

            // Prefill in chunks so prompts longer than 512 tokens (class-plan
            // pre-loads whole files) don't silently overflow the batch.
            let prefill_chunk: usize = 512;
            let mut batch = LlamaBatch::new(prefill_chunk, 1);
            let total = tokens_list.len();
            let mut idx: usize = 0;
            while idx < total {
                let chunk_end = (idx + prefill_chunk).min(total);
                batch.clear();
                let mut ok = true;
                for j in idx..chunk_end {
                    let pos = j as i32;
                    let is_last = j + 1 == total;
                    if batch.add(tokens_list[j], pos, &[0], is_last).is_err() {
                        ok = false;
                        break;
                    }
                }
                if !ok || ctx.decode(&mut batch).is_err() {
                    let _ = tx.blocking_send(TokenEvent::Done {
                        stop_reason: StopReason::BackendAborted,
                        usage: Usage::default(),
                    });
                    return;
                }
                idx = chunk_end;
            }

            let mut n_cur = total as i32;
            let mut sampler = LlamaSampler::chain_simple([
                LlamaSampler::temp(temperature),
                LlamaSampler::top_k(40),
                LlamaSampler::top_p(0.95, 1),
                LlamaSampler::greedy(),
            ]);

            let mut produced: u32 = 0;
            let mut stop_reason = StopReason::MaxTokens;
            while (produced as usize) < max_tokens {
                let token: LlamaToken = sampler.sample(&ctx, batch.n_tokens() - 1);
                sampler.accept(token);
                if model.is_eog_token(token) {
                    stop_reason = StopReason::Eos;
                    break;
                }
                let piece_bytes = match model.token_to_bytes(token, llama_cpp_2::model::Special::Tokenize) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let piece = String::from_utf8_lossy(&piece_bytes).into_owned();
                if tx
                    .blocking_send(TokenEvent::Delta { text: piece })
                    .is_err()
                {
                    return;
                }
                produced += 1;

                batch.clear();
                if batch.add(token, n_cur, &[0], true).is_err() {
                    break;
                }
                n_cur += 1;
                if ctx.decode(&mut batch).is_err() {
                    break;
                }
            }
            let _ = tx.blocking_send(TokenEvent::Done {
                stop_reason,
                usage: Usage {
                    prompt_tokens: tokens_list.len() as u32,
                    completion_tokens: produced,
                    thinking_tokens: 0,
                },
            });
        });

        Ok(tokio_stream::wrappers::ReceiverStream::new(rx).boxed())
    }
}

/// Render the chat history into a single prompt string. We use the Gemma
/// instruction-tuned chat template: `<start_of_turn>role\n...<end_of_turn>\n`.
fn render_chat_prompt(req: &GenerateRequest) -> String {
    let mut s = String::new();
    for m in &req.messages {
        // Gemma 4 has native `system` role support; earlier Gemma generations
        // did not. Map System messages to "system" — llama.cpp's Gemma 4 chat
        // template handles it; if Gemma collapses it back to a user turn
        // internally, behaviour matches Gemma 3 anyway.
        let tag = match m.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "model",
            MessageRole::Tool => "user", // append tool results as user observations
        };
        s.push_str("<start_of_turn>");
        s.push_str(tag);
        s.push('\n');
        if matches!(m.role, MessageRole::Tool) {
            if let Some(name) = &m.tool_name {
                s.push_str("(tool ");
                s.push_str(name);
                s.push_str(" result)\n");
            }
        }
        s.push_str(&m.content);
        if !m.tool_calls.is_empty() {
            for tc in &m.tool_calls {
                s.push('\n');
                s.push_str(&format!(
                    "{{\"name\":\"{}\",\"args\":{}}}",
                    tc.name,
                    serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".into())
                ));
            }
        }
        s.push_str("<end_of_turn>\n");
    }
    s.push_str("<start_of_turn>model\n");
    s
}
