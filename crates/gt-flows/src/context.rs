use chrono::NaiveDate;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::artifacts::ArtifactMap;

/// Per-flow execution context. Owned by the orchestrator and mutated as each
/// step finishes.
pub struct FlowCtx {
    pub root: PathBuf,
    pub date: NaiveDate,
    /// Free-form per-flow inputs (e.g. {"name": "Maya", "description": "..."}).
    pub inputs: BTreeMap<String, String>,
    /// Artifacts produced by completed steps.
    pub artifacts: ArtifactMap,
}

impl FlowCtx {
    pub fn new(root: impl AsRef<Path>, date: NaiveDate) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            date,
            inputs: BTreeMap::new(),
            artifacts: ArtifactMap::new(),
        }
    }
    pub fn with_input(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.inputs.insert(key.into(), value.into());
        self
    }
}
