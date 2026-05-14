use async_trait::async_trait;
use gt_core::tool::{ParameterKind, ParameterSchema, Tool, ToolCtx, ToolResult, ToolSchema};

use crate::path::resolve_scoped;

const DEFAULT_MAX_BYTES: usize = 200_000;
const HARD_CAP_BYTES: usize = 1_000_000; // 1MB

pub struct ReadTool;

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &'static str {
        "Read"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "Read".into(),
            description: "Read a UTF-8 text file inside the session working directory.".into(),
            parameters: vec![
                ParameterSchema {
                    name: "path".into(),
                    kind: ParameterKind::String,
                    required: true,
                    description: "path relative to the working directory".into(),
                },
                ParameterSchema {
                    name: "max_bytes".into(),
                    kind: ParameterKind::Integer,
                    required: false,
                    description: "truncate at this many bytes (default 200000)".into(),
                },
            ],
        }
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolCtx<'_>) -> ToolResult {
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolResult::err("missing required argument `path`."),
        };
        let max_bytes = args
            .get("max_bytes")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_MAX_BYTES)
            .min(HARD_CAP_BYTES);

        let target = match resolve_scoped(ctx.working_dir, &path) {
            Ok(p) => p,
            Err(e) => return ToolResult::err(format!("{e}")),
        };

        if !target.exists() {
            let listing = list_dir_one_level(ctx.working_dir);
            return ToolResult::err(format!(
                "File '{path}' does not exist.\nContents of working directory:\n{listing}"
            ));
        }
        if !target.is_file() {
            return ToolResult::err(format!("'{path}' is not a regular file."));
        }

        match tokio::fs::read(&target).await {
            Ok(bytes) => {
                let truncated = bytes.len() > max_bytes;
                let slice: &[u8] = if truncated { &bytes[..max_bytes] } else { &bytes };
                match std::str::from_utf8(slice) {
                    Ok(s) => {
                        if truncated {
                            ToolResult::ok(format!("{s}\n\n[truncated at {max_bytes} bytes]"))
                        } else {
                            ToolResult::ok(s.to_string())
                        }
                    }
                    Err(_) => ToolResult::err(format!("'{path}' is not valid UTF-8.")),
                }
            }
            Err(e) => ToolResult::err(format!("read error on '{path}': {e}")),
        }
    }
}

fn list_dir_one_level(dir: &std::path::Path) -> String {
    let mut out = String::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let kind = if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                "/"
            } else {
                ""
            };
            out.push_str(&format!("  {name}{kind}\n"));
        }
    }
    if out.is_empty() {
        out = "  (empty)".into();
    }
    out
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
    async fn reads_existing_file() {
        let dir = tempdir().unwrap();
        tokio::fs::write(dir.path().join("a.md"), "hello").await.unwrap();
        let r = ReadTool
            .execute(serde_json::json!({"path":"a.md"}), &ctx(dir.path()))
            .await;
        assert!(!r.is_error);
        assert_eq!(r.output, "hello");
    }

    #[tokio::test]
    async fn rejects_absolute_path() {
        let dir = tempdir().unwrap();
        let r = ReadTool
            .execute(serde_json::json!({"path":"/etc/passwd"}), &ctx(dir.path()))
            .await;
        assert!(r.is_error);
        assert!(r.output.to_lowercase().contains("absolute"));
    }

    #[tokio::test]
    async fn rejects_dotdot_escape() {
        let dir = tempdir().unwrap();
        let r = ReadTool
            .execute(serde_json::json!({"path":"../x"}), &ctx(dir.path()))
            .await;
        assert!(r.is_error);
    }

    #[tokio::test]
    async fn missing_file_lists_directory_contents() {
        let dir = tempdir().unwrap();
        tokio::fs::write(dir.path().join("other.md"), "x").await.unwrap();
        let r = ReadTool
            .execute(serde_json::json!({"path":"missing.md"}), &ctx(dir.path()))
            .await;
        assert!(r.is_error);
        assert!(r.output.contains("other.md"));
    }

    #[tokio::test]
    async fn truncates_at_max_bytes() {
        let dir = tempdir().unwrap();
        let big = "a".repeat(10_000);
        tokio::fs::write(dir.path().join("big.md"), &big).await.unwrap();
        let r = ReadTool
            .execute(
                serde_json::json!({"path":"big.md","max_bytes":100}),
                &ctx(dir.path()),
            )
            .await;
        assert!(!r.is_error);
        assert!(r.output.contains("[truncated at 100 bytes]"));
    }
}
