use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::ids::SessionId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterKind {
    String,
    Integer,
    Number,
    Boolean,
    Object,
    Array,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSchema {
    pub name: String,
    pub kind: ParameterKind,
    pub required: bool,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ParameterSchema>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub output: String,
    pub is_error: bool,
    /// Optional structured suggestion the parser/quality monitor can surface to the user
    /// (e.g. the literal recovery JSON shape from Write-Guard).
    pub suggestion: Option<String>,
}

impl ToolResult {
    pub fn ok(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            is_error: false,
            suggestion: None,
        }
    }
    pub fn err(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            is_error: true,
            suggestion: None,
        }
    }
    pub fn err_with_suggestion(output: impl Into<String>, suggestion: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            is_error: true,
            suggestion: Some(suggestion.into()),
        }
    }
}

/// Context passed to every tool execution.
pub struct ToolCtx<'a> {
    pub working_dir: &'a Path,
    pub session_id: SessionId,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn schema(&self) -> ToolSchema;
    async fn execute(&self, args: serde_json::Value, ctx: &ToolCtx<'_>) -> ToolResult;
}

/// In-memory registry keyed by tool name.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
    working_dir: Option<PathBuf>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_working_dir(mut self, dir: PathBuf) -> Self {
        self.working_dir = Some(dir);
        self
    }

    pub fn working_dir(&self) -> Option<&Path> {
        self.working_dir.as_deref()
    }

    pub fn register(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.insert(tool.name().to_string(), tool);
        self
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub fn schemas(&self) -> Vec<ToolSchema> {
        self.tools.values().map(|t| t.schema()).collect()
    }

    /// Build a new registry containing only the named subset of tools.
    /// Names that do not exist in `self` are silently dropped (the caller's
    /// authoritative list of allowed tools is what we want to honor).
    pub fn allowed_subset(&self, names: &[String]) -> Self {
        let tools = names
            .iter()
            .filter_map(|n| self.tools.get(n).map(|t| (n.clone(), t.clone())))
            .collect();
        Self {
            tools,
            working_dir: self.working_dir.clone(),
        }
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::SessionId;

    struct StubTool {
        name_: &'static str,
    }

    #[async_trait]
    impl Tool for StubTool {
        fn name(&self) -> &'static str {
            self.name_
        }
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: self.name_.into(),
                description: "stub".into(),
                parameters: vec![],
            }
        }
        async fn execute(&self, _args: serde_json::Value, _ctx: &ToolCtx<'_>) -> ToolResult {
            ToolResult::ok("stub")
        }
    }

    fn reg() -> ToolRegistry {
        ToolRegistry::new()
            .register(Arc::new(StubTool { name_: "Read" }))
            .register(Arc::new(StubTool { name_: "Write" }))
            .register(Arc::new(StubTool { name_: "Edit" }))
    }

    #[test]
    fn allowed_subset_filters_to_named_tools() {
        let r = reg().allowed_subset(&["Read".into(), "Write".into()]);
        assert_eq!(r.len(), 2);
        assert!(r.get("Read").is_some());
        assert!(r.get("Write").is_some());
        assert!(r.get("Edit").is_none());
    }

    #[test]
    fn allowed_subset_drops_unknown_names() {
        let r = reg().allowed_subset(&["Read".into(), "DoesNotExist".into()]);
        assert_eq!(r.len(), 1);
        assert!(r.get("Read").is_some());
    }

    #[test]
    fn allowed_subset_preserves_working_dir() {
        let r = reg()
            .with_working_dir(PathBuf::from("/tmp/wd"))
            .allowed_subset(&["Read".into()]);
        assert_eq!(r.working_dir(), Some(Path::new("/tmp/wd")));
    }

    #[tokio::test]
    async fn tool_executes_through_registry() {
        let r = reg();
        let tool = r.get("Read").unwrap();
        let ctx = ToolCtx {
            working_dir: Path::new("/tmp"),
            session_id: SessionId::new(),
        };
        let out = tool.execute(serde_json::json!({}), &ctx).await;
        assert!(!out.is_error);
        assert_eq!(out.output, "stub");
    }
}
