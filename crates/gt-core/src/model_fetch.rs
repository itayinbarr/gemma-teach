//! Resumable HuggingFace GGUF download with SHA-256 verification.
//!
//! Only available when the `model-fetch` feature is enabled. Progress events
//! are emitted through a tokio `mpsc::Sender` so the TUI can render a live
//! progress bar.

#![cfg(feature = "model-fetch")]

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;

#[derive(Debug, Error)]
pub enum FetchError {
    #[error("http error: {0}")]
    Http(String),
    #[error("io error on '{0}': {1}")]
    Io(String, String),
    #[error("hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FetchEvent {
    Starting { total_bytes: Option<u64> },
    Progress { downloaded: u64, total: Option<u64> },
    Verifying,
    Done,
    Failed { error: String },
}

#[derive(Debug, Clone)]
pub struct FetchSpec {
    pub url: String,
    pub dest: PathBuf,
    /// Lowercase hex SHA-256 digest. If `None`, verification is skipped.
    pub expected_sha256: Option<String>,
}

impl FetchSpec {
    pub fn gemma_4_e2b_q4km(default_dir: &Path) -> Self {
        Self {
            url: "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"
                .into(),
            dest: default_dir.join("gemma-4-E2B-it-Q4_K_M.gguf"),
            expected_sha256: None,
        }
    }
}

pub async fn fetch(
    spec: FetchSpec,
    events: Option<mpsc::Sender<FetchEvent>>,
) -> Result<PathBuf, FetchError> {
    let emit = |e: FetchEvent| {
        if let Some(tx) = &events {
            let tx = tx.clone();
            tokio::spawn(async move {
                let _ = tx.send(e).await;
            });
        }
    };

    if spec.dest.exists() {
        emit(FetchEvent::Done);
        return Ok(spec.dest);
    }

    if let Some(parent) = spec.dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| FetchError::Io(parent.display().to_string(), e.to_string()))?;
    }

    let partial = spec.dest.with_extension("gguf.partial");
    let mut already: u64 = 0;
    if partial.exists() {
        already = tokio::fs::metadata(&partial)
            .await
            .map(|m| m.len())
            .unwrap_or(0);
    }

    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| FetchError::Http(e.to_string()))?;
    let mut req = client.get(&spec.url);
    if already > 0 {
        req = req.header("Range", format!("bytes={}-", already));
    }
    let resp = req.send().await.map_err(|e| FetchError::Http(e.to_string()))?;
    if !resp.status().is_success() && resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(FetchError::Http(format!("status {}", resp.status())));
    }
    let total = resp
        .content_length()
        .map(|cl| already + cl)
        .or(resp.content_length());
    emit(FetchEvent::Starting { total_bytes: total });

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&partial)
        .await
        .map_err(|e| FetchError::Io(partial.display().to_string(), e.to_string()))?;

    let mut downloaded = already;
    let mut stream = resp.bytes_stream();
    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(|e| FetchError::Http(e.to_string()))?;
        file.write_all(&bytes)
            .await
            .map_err(|e| FetchError::Io(partial.display().to_string(), e.to_string()))?;
        downloaded += bytes.len() as u64;
        emit(FetchEvent::Progress { downloaded, total });
    }
    file.flush()
        .await
        .map_err(|e| FetchError::Io(partial.display().to_string(), e.to_string()))?;
    drop(file);

    // Verify if a hash was provided.
    if let Some(expected) = &spec.expected_sha256 {
        emit(FetchEvent::Verifying);
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        let mut f = tokio::fs::File::open(&partial)
            .await
            .map_err(|e| FetchError::Io(partial.display().to_string(), e.to_string()))?;
        let mut buf = vec![0u8; 1 << 16];
        use tokio::io::AsyncReadExt;
        loop {
            let n = f
                .read(&mut buf)
                .await
                .map_err(|e| FetchError::Io(partial.display().to_string(), e.to_string()))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let actual = hex::encode(hasher.finalize());
        if &actual != expected {
            let _ = tokio::fs::remove_file(&partial).await;
            emit(FetchEvent::Failed {
                error: format!("hash mismatch (expected {expected}, got {actual})"),
            });
            return Err(FetchError::HashMismatch {
                expected: expected.clone(),
                actual,
            });
        }
    }

    tokio::fs::rename(&partial, &spec.dest)
        .await
        .map_err(|e| FetchError::Io(spec.dest.display().to_string(), e.to_string()))?;
    emit(FetchEvent::Done);
    Ok(spec.dest)
}

/// Default cache directory: `~/.gemma-teach/models`. Caller is responsible for
/// honoring `$GEMMA_TEACH_MODEL`.
pub fn default_models_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".gemma-teach").join("models")
    } else {
        PathBuf::from(".gemma-teach/models")
    }
}
