use async_trait::async_trait;
use gt_core::tool::{ParameterKind, ParameterSchema, Tool, ToolCtx, ToolResult, ToolSchema};

use crate::path::resolve_scoped;

pub struct WriteTool;

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &'static str {
        "Write"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "Write".into(),
            description: "Create a NEW file inside the working directory. Refuses if it exists."
                .into(),
            parameters: vec![
                ParameterSchema {
                    name: "path".into(),
                    kind: ParameterKind::String,
                    required: true,
                    description: "path relative to working directory".into(),
                },
                ParameterSchema {
                    name: "content".into(),
                    kind: ParameterKind::String,
                    required: true,
                    description: "full file content".into(),
                },
            ],
        }
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolCtx<'_>) -> ToolResult {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolResult::err("missing required argument `path`."),
        };
        let content = match args.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => return ToolResult::err("missing required argument `content`."),
        };

        let target = match resolve_scoped(ctx.working_dir, &path) {
            Ok(p) => p,
            Err(e) => return ToolResult::err(format!("{e}")),
        };

        if target.exists() {
            // Write-Guard. The recovery instruction MUST contain the literal
            // Edit JSON shape for this exact path — this is non-negotiable.
            let recipe = format!(
                "Error: Write refused — '{path}' already exists. Write only creates NEW files.\n\nTo change the existing file, use Edit:\n{{\"name\":\"Edit\",\"args\":{{\"path\":\"{path}\",\"old_text\":\"<exact text currently in the file>\",\"new_text\":\"<replacement text>\"}}}}\n\nIf you do not know the file's current content, Read it first. For several changes, emit several Edit calls — one per location. Do NOT retry Write; it will refuse again."
            );
            return ToolResult::err_with_suggestion(recipe.clone(), recipe);
        }

        if let Some(parent) = target.parent() {
            if !parent.exists() {
                if let Err(e) = tokio::fs::create_dir_all(parent).await {
                    return ToolResult::err(format!("could not create parent of '{path}': {e}"));
                }
            }
        }
        if let Err(e) = tokio::fs::write(&target, &content).await {
            return ToolResult::err(format!("write error on '{path}': {e}"));
        }
        ToolResult::ok(format!(
            "Wrote {} bytes to {}",
            content.len(),
            path
        ))
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
    async fn creates_new_file() {
        let dir = tempdir().unwrap();
        let r = WriteTool
            .execute(
                serde_json::json!({"path":"a.md","content":"hello"}),
                &ctx(dir.path()),
            )
            .await;
        assert!(!r.is_error);
        let got = tokio::fs::read_to_string(dir.path().join("a.md")).await.unwrap();
        assert_eq!(got, "hello");
    }

    #[tokio::test]
    async fn write_guard_refuses_existing_with_edit_recipe() {
        let dir = tempdir().unwrap();
        tokio::fs::write(dir.path().join("a.md"), "x").await.unwrap();
        let r = WriteTool
            .execute(
                serde_json::json!({"path":"a.md","content":"new"}),
                &ctx(dir.path()),
            )
            .await;
        assert!(r.is_error);
        // verbatim shape — locking it down so future refactors don't soften the recipe.
        assert!(r.output.contains("\"name\":\"Edit\""));
        assert!(r.output.contains("\"path\":\"a.md\""));
        assert!(r.output.contains("\"old_text\""));
        assert!(r.output.contains("\"new_text\""));
        // unchanged on disk
        let got = tokio::fs::read_to_string(dir.path().join("a.md")).await.unwrap();
        assert_eq!(got, "x");
    }

    #[tokio::test]
    async fn creates_parent_dirs_if_missing() {
        let dir = tempdir().unwrap();
        let r = WriteTool
            .execute(
                serde_json::json!({"path":"sub/dir/a.md","content":"k"}),
                &ctx(dir.path()),
            )
            .await;
        assert!(!r.is_error);
        let got = tokio::fs::read_to_string(dir.path().join("sub/dir/a.md")).await.unwrap();
        assert_eq!(got, "k");
    }
}
