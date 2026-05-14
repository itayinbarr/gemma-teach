use async_trait::async_trait;
use gt_core::tool::{ParameterKind, ParameterSchema, Tool, ToolCtx, ToolResult, ToolSchema};

use crate::path::resolve_scoped;

pub struct EditTool;

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &'static str {
        "Edit"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "Edit".into(),
            description: "Replace an exact text block in an existing file inside the working dir."
                .into(),
            parameters: vec![
                ParameterSchema {
                    name: "path".into(),
                    kind: ParameterKind::String,
                    required: true,
                    description: "path relative to working directory".into(),
                },
                ParameterSchema {
                    name: "old_text".into(),
                    kind: ParameterKind::String,
                    required: true,
                    description: "exact text currently in the file".into(),
                },
                ParameterSchema {
                    name: "new_text".into(),
                    kind: ParameterKind::String,
                    required: true,
                    description: "replacement text".into(),
                },
            ],
        }
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolCtx<'_>) -> ToolResult {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => return ToolResult::err("missing required argument `path`."),
        };
        let old_text = match args.get("old_text").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => return ToolResult::err("missing required argument `old_text`."),
        };
        let new_text = match args.get("new_text").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => return ToolResult::err("missing required argument `new_text`."),
        };

        if old_text.is_empty() {
            return ToolResult::err(
                "`old_text` is empty. Edit requires the exact text to replace.",
            );
        }

        let target = match resolve_scoped(ctx.working_dir, &path) {
            Ok(p) => p,
            Err(e) => return ToolResult::err(format!("{e}")),
        };
        if !target.exists() {
            return ToolResult::err(format!(
                "File '{path}' does not exist. Use Write to create a new file."
            ));
        }
        if !target.is_file() {
            return ToolResult::err(format!("'{path}' is not a regular file."));
        }

        let content = match tokio::fs::read_to_string(&target).await {
            Ok(s) => s,
            Err(e) => return ToolResult::err(format!("read error on '{path}': {e}")),
        };

        let matches: Vec<_> = content.match_indices(&old_text).collect();
        if matches.is_empty() {
            return ToolResult::err(format!(
                "No match for `old_text` in '{path}'. Read the file first and copy the exact text (including whitespace). Surrounding context may have changed."
            ));
        }
        if matches.len() > 1 {
            return ToolResult::err(format!(
                "`old_text` matches {} locations in '{path}'. Include more surrounding context to make it unique.",
                matches.len()
            ));
        }
        let (idx, _) = matches[0];
        let mut new_content = String::with_capacity(content.len() + new_text.len());
        new_content.push_str(&content[..idx]);
        new_content.push_str(&new_text);
        new_content.push_str(&content[idx + old_text.len()..]);

        if let Err(e) = tokio::fs::write(&target, &new_content).await {
            return ToolResult::err(format!("write error on '{path}': {e}"));
        }
        ToolResult::ok(format!("Edited '{path}' (1 replacement)."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gt_core::ids::SessionId;
    use tempfile::tempdir;

    fn ctx<'a>(wd: &'a std::path::Path) -> ToolCtx<'a> {
        ToolCtx {
            working_dir: wd,
            session_id: SessionId::new(),
        }
    }

    #[tokio::test]
    async fn replaces_unique_match() {
        let dir = tempdir().unwrap();
        tokio::fs::write(dir.path().join("a.md"), "hello world").await.unwrap();
        let r = EditTool
            .execute(
                serde_json::json!({"path":"a.md","old_text":"world","new_text":"there"}),
                &ctx(dir.path()),
            )
            .await;
        assert!(!r.is_error, "{}", r.output);
        let got = tokio::fs::read_to_string(dir.path().join("a.md")).await.unwrap();
        assert_eq!(got, "hello there");
    }

    #[tokio::test]
    async fn refuses_when_no_match() {
        let dir = tempdir().unwrap();
        tokio::fs::write(dir.path().join("a.md"), "hello world").await.unwrap();
        let r = EditTool
            .execute(
                serde_json::json!({"path":"a.md","old_text":"missing","new_text":"x"}),
                &ctx(dir.path()),
            )
            .await;
        assert!(r.is_error);
        assert!(r.output.to_lowercase().contains("no match"));
    }

    #[tokio::test]
    async fn refuses_when_multiple_matches() {
        let dir = tempdir().unwrap();
        tokio::fs::write(dir.path().join("a.md"), "ab ab ab").await.unwrap();
        let r = EditTool
            .execute(
                serde_json::json!({"path":"a.md","old_text":"ab","new_text":"X"}),
                &ctx(dir.path()),
            )
            .await;
        assert!(r.is_error);
        assert!(r.output.contains("matches"));
    }

    #[tokio::test]
    async fn refuses_when_file_missing() {
        let dir = tempdir().unwrap();
        let r = EditTool
            .execute(
                serde_json::json!({"path":"missing.md","old_text":"x","new_text":"y"}),
                &ctx(dir.path()),
            )
            .await;
        assert!(r.is_error);
        assert!(r.output.contains("does not exist"));
    }
}
