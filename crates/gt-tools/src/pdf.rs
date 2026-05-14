//! PDF compile runner. Deterministic flow step (no model).

use async_trait::async_trait;
use std::path::Path;
use thiserror::Error;
use tokio::process::Command;

#[derive(Debug, Error)]
pub enum PdfError {
    #[error("typst spawn failed: {0}")]
    Spawn(String),
    #[error("typst failed (exit={code}, stderr={stderr})")]
    Failed { code: i32, stderr: String },
    #[error("missing dependency: typst. Install via `brew install typst`")]
    Missing,
}

#[async_trait]
pub trait PdfRunner: Send + Sync {
    async fn compile(
        &self,
        md_path: &Path,
        template_path: &Path,
        out_pdf: &Path,
    ) -> Result<(), PdfError>;
}

/// Typst-based PDF runner. We write a tiny shim Typst file that:
///   1. Sets the document metadata via the chosen template.
///   2. Includes the Markdown content rendered as Typst-flavored markup.
///
/// The shim is created in a temp dir alongside the source `.md` so Typst
/// can resolve any local paths.
pub struct TypstRunner;

impl TypstRunner {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TypstRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PdfRunner for TypstRunner {
    async fn compile(
        &self,
        md_path: &Path,
        template_path: &Path,
        out_pdf: &Path,
    ) -> Result<(), PdfError> {
        if which::which("typst").is_err() {
            return Err(PdfError::Missing);
        }

        let md_text = tokio::fs::read_to_string(md_path)
            .await
            .map_err(|e| PdfError::Spawn(format!("read md: {e}")))?;

        // Build a Typst entrypoint that imports the template and renders the body.
        // We render Markdown by converting it inline — Typst's native markdown
        // import isn't universally available, so we use a small heuristic
        // conversion to Typst markup.
        let typst_src = format!(
            r#"#import "{template}" as template
#show: template.notes.with(title: "{title}")

{body}
"#,
            template = template_path.display(),
            title = filename_to_title(md_path),
            body = md_to_typst(&md_text),
        );

        let tmp = tempfile::tempdir().map_err(|e| PdfError::Spawn(e.to_string()))?;
        let entry = tmp.path().join("entry.typ");
        tokio::fs::write(&entry, typst_src.as_bytes())
            .await
            .map_err(|e| PdfError::Spawn(format!("write entry: {e}")))?;

        if let Some(parent) = out_pdf.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| PdfError::Spawn(format!("mkdir: {e}")))?;
        }

        let status = Command::new("typst")
            .arg("compile")
            .arg(&entry)
            .arg(out_pdf)
            .output()
            .await
            .map_err(|e| PdfError::Spawn(format!("spawn: {e}")))?;
        if !status.status.success() {
            return Err(PdfError::Failed {
                code: status.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&status.stderr).into_owned(),
            });
        }
        Ok(())
    }
}

fn filename_to_title(p: &Path) -> String {
    p.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.replace('-', " ").replace('_', " "))
        .unwrap_or_else(|| "Document".into())
}

/// Tiny markdown-to-Typst conversion. Handles headings, list bullets, and
/// numbered lists; everything else passes through as plain text. Good enough
/// for our class-notes / homework output; we can move to a richer converter
/// later if traces show issues.
fn md_to_typst(md: &str) -> String {
    let mut out = String::with_capacity(md.len() + 64);
    for line in md.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("# ") {
            out.push_str(&format!("= {}\n", escape_typst(rest)));
        } else if let Some(rest) = trimmed.strip_prefix("## ") {
            out.push_str(&format!("== {}\n", escape_typst(rest)));
        } else if let Some(rest) = trimmed.strip_prefix("### ") {
            out.push_str(&format!("=== {}\n", escape_typst(rest)));
        } else if let Some(rest) = trimmed.strip_prefix("- ") {
            out.push_str(&format!("- {}\n", escape_typst(rest)));
        } else if let Some(rest) = trimmed.strip_prefix("* ") {
            out.push_str(&format!("- {}\n", escape_typst(rest)));
        } else if let Some((num, rest)) = numbered_prefix(trimmed) {
            out.push_str(&format!("+ {} {}\n", num, escape_typst(rest)));
        } else {
            out.push_str(&escape_typst(line));
            out.push('\n');
        }
    }
    out
}

fn numbered_prefix(s: &str) -> Option<(String, &str)> {
    let mut chars = s.char_indices();
    let mut idx = 0;
    let mut saw_digit = false;
    while let Some((i, c)) = chars.next() {
        if c.is_ascii_digit() {
            saw_digit = true;
            idx = i + c.len_utf8();
            continue;
        }
        if saw_digit && c == '.' {
            // optional space
            let rest = &s[idx + c.len_utf8()..];
            let rest = rest.strip_prefix(' ').unwrap_or(rest);
            let num = s[..idx].to_string();
            return Some((num, rest));
        }
        break;
    }
    None
}

fn escape_typst(s: &str) -> String {
    s.replace('#', "\\#").replace('@', "\\@")
}

/// Mock for tests: writes a stub PDF (actually plain text) so flows don't need
/// the real typst binary in CI.
pub struct MockPdfRunner;

#[async_trait]
impl PdfRunner for MockPdfRunner {
    async fn compile(
        &self,
        md_path: &Path,
        _template_path: &Path,
        out_pdf: &Path,
    ) -> Result<(), PdfError> {
        let md = tokio::fs::read_to_string(md_path)
            .await
            .map_err(|e| PdfError::Spawn(format!("read md: {e}")))?;
        if let Some(parent) = out_pdf.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| PdfError::Spawn(format!("mkdir: {e}")))?;
        }
        tokio::fs::write(out_pdf, format!("MOCK PDF\n\n{md}").as_bytes())
            .await
            .map_err(|e| PdfError::Spawn(format!("write: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn md_headings_convert() {
        let out = md_to_typst("# Title\n## Sub\n### Sub2\n- bullet\n1. one\n");
        assert!(out.contains("= Title"));
        assert!(out.contains("== Sub"));
        assert!(out.contains("=== Sub2"));
        assert!(out.contains("- bullet"));
        assert!(out.contains("+ 1 one"));
    }
}
