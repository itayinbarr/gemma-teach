//! Backend factory + first-launch model download.

use anyhow::Result;
#[cfg(feature = "llama")]
use anyhow::Context;
use gt_core::backend::{EchoBackend, LlmBackend, MockBackend};
use std::path::PathBuf;
use std::sync::Arc;

pub async fn build(name: &str, model_path: Option<PathBuf>) -> Result<Arc<dyn LlmBackend>> {
    match name {
        "mock" => Ok(Arc::new(MockBackend::new())),
        "echo" => Ok(Arc::new(EchoBackend::new())),
        "llama" => build_llama(model_path).await,
        other => anyhow::bail!(
            "unknown --backend '{}'. valid: llama | mock | echo",
            other
        ),
    }
}

#[cfg(feature = "llama")]
async fn build_llama(model_path: Option<PathBuf>) -> Result<Arc<dyn LlmBackend>> {
    use gt_core::llama_backend::{LlamaCppBackend, LlamaConfig};
    use gt_core::model_fetch::default_models_dir;
    let path = match model_path {
        Some(p) => p,
        None => {
            let dir = default_models_dir();
            let candidate = dir.join("gemma-4-E2B-it-Q4_K_M.gguf");
            if !candidate.exists() {
                download_default_model().await?;
            }
            candidate
        }
    };
    if !path.exists() {
        anyhow::bail!(
            "model not found at {}. Run `gemma-teach --download-only` or set --model.",
            path.display()
        );
    }
    Ok(Arc::new(LlamaCppBackend::new(LlamaConfig::new(path))))
}

#[cfg(not(feature = "llama"))]
async fn build_llama(_model_path: Option<PathBuf>) -> Result<Arc<dyn LlmBackend>> {
    anyhow::bail!(
        "the llama backend is not compiled into this binary. Rebuild with `cargo build --features llama` or pass --backend mock"
    )
}

#[cfg(feature = "llama")]
pub async fn download_default_model() -> Result<()> {
    use gt_core::model_fetch::{default_models_dir, fetch, FetchSpec};
    let dir = default_models_dir();
    tokio::fs::create_dir_all(&dir).await.ok();
    let spec = FetchSpec::gemma_4_e2b_q4km(&dir);

    eprintln!("Downloading Gemma 4 E2B (Q4_K_M) — this is ~3.1 GB on first run.");
    eprintln!("Destination: {}", spec.dest.display());

    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let print_handle = tokio::spawn(async move {
        let mut last_pct = -1i32;
        while let Some(ev) = rx.recv().await {
            use gt_core::model_fetch::FetchEvent;
            match ev {
                FetchEvent::Starting { total_bytes } => {
                    if let Some(t) = total_bytes {
                        eprintln!("total bytes: {}", t);
                    }
                }
                FetchEvent::Progress { downloaded, total } => {
                    if let Some(t) = total {
                        let pct = ((downloaded as f64 / t as f64) * 100.0) as i32;
                        if pct != last_pct {
                            eprint!("\r{:>3}%  ({} / {} bytes)", pct, downloaded, t);
                            use std::io::Write;
                            std::io::stderr().flush().ok();
                            last_pct = pct;
                        }
                    } else {
                        eprint!("\r{} bytes", downloaded);
                        use std::io::Write;
                        std::io::stderr().flush().ok();
                    }
                }
                FetchEvent::Verifying => eprintln!("\nverifying..."),
                FetchEvent::Done => eprintln!("\ndone."),
                FetchEvent::Failed { error } => eprintln!("\nfailed: {error}"),
            }
        }
    });
    fetch(spec, Some(tx))
        .await
        .context("model download failed")?;
    print_handle.await.ok();
    Ok(())
}

#[cfg(not(feature = "llama"))]
pub async fn download_default_model() -> Result<()> {
    anyhow::bail!("model-fetch feature not compiled in. Rebuild with --features llama.")
}
