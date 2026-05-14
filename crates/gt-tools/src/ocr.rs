//! OCR runner. Not a model tool — used by deterministic flow steps.

use async_trait::async_trait;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tokio::process::Command;

#[derive(Debug, Error)]
pub enum OcrError {
    #[error("OCR runner spawn failed: {0}")]
    Spawn(String),
    #[error("OCR runner failed (exit={code}, stderr={stderr})")]
    Failed { code: i32, stderr: String },
    #[error("missing dependency: {0}. Install via `brew install {1}`")]
    Missing(&'static str, &'static str),
}

#[async_trait]
pub trait OcrRunner: Send + Sync {
    async fn ocr_pdf_to_text(&self, pdf: &Path, out_txt: &Path) -> Result<(), OcrError>;
}

pub struct TesseractRunner {
    pub dpi: u32,
}

impl TesseractRunner {
    pub fn new() -> Self {
        Self { dpi: 300 }
    }
}

impl Default for TesseractRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OcrRunner for TesseractRunner {
    async fn ocr_pdf_to_text(&self, pdf: &Path, out_txt: &Path) -> Result<(), OcrError> {
        if which::which("pdftoppm").is_err() {
            return Err(OcrError::Missing("pdftoppm", "poppler"));
        }
        if which::which("tesseract").is_err() {
            return Err(OcrError::Missing("tesseract", "tesseract"));
        }
        let tmp = tempfile::tempdir().map_err(|e| OcrError::Spawn(e.to_string()))?;
        let prefix = tmp.path().join("page");

        // pdftoppm -r <dpi> <pdf> <prefix> -png
        let status = Command::new("pdftoppm")
            .arg("-r")
            .arg(self.dpi.to_string())
            .arg(pdf)
            .arg(&prefix)
            .arg("-png")
            .output()
            .await
            .map_err(|e| OcrError::Spawn(format!("pdftoppm: {e}")))?;
        if !status.status.success() {
            return Err(OcrError::Failed {
                code: status.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&status.stderr).into_owned(),
            });
        }

        // Collect produced pages in lexical order.
        let mut pages: Vec<PathBuf> = walkdir::WalkDir::new(tmp.path())
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
            .map(|e| e.into_path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
            .collect();
        pages.sort();

        let mut combined = String::new();
        for page in &pages {
            let out = Command::new("tesseract")
                .arg(page)
                .arg("stdout")
                .output()
                .await
                .map_err(|e| OcrError::Spawn(format!("tesseract: {e}")))?;
            if !out.status.success() {
                return Err(OcrError::Failed {
                    code: out.status.code().unwrap_or(-1),
                    stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
                });
            }
            combined.push_str(&String::from_utf8_lossy(&out.stdout));
            combined.push_str("\n\n");
        }
        if let Some(parent) = out_txt.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| OcrError::Spawn(format!("mkdir: {e}")))?;
        }
        tokio::fs::write(out_txt, combined.as_bytes())
            .await
            .map_err(|e| OcrError::Spawn(format!("write {}: {e}", out_txt.display())))?;
        Ok(())
    }
}

/// Mock implementation for tests: writes a caller-provided string straight to `out_txt`.
pub struct MockOcrRunner {
    pub text: String,
}

#[async_trait]
impl OcrRunner for MockOcrRunner {
    async fn ocr_pdf_to_text(&self, _pdf: &Path, out_txt: &Path) -> Result<(), OcrError> {
        if let Some(parent) = out_txt.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| OcrError::Spawn(format!("mkdir: {e}")))?;
        }
        tokio::fs::write(out_txt, self.text.as_bytes())
            .await
            .map_err(|e| OcrError::Spawn(format!("write: {e}")))?;
        Ok(())
    }
}
