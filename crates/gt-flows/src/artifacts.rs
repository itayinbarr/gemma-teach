use std::collections::BTreeMap;
use std::path::PathBuf;

pub type ArtifactKey = String;

/// Resolved artifacts produced by prior steps in the same flow run.
/// Keys are flow-defined symbolic names (e.g. "student_md", "tags_json").
#[derive(Debug, Default, Clone)]
pub struct ArtifactMap {
    inner: BTreeMap<ArtifactKey, PathBuf>,
}

impl ArtifactMap {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(&mut self, key: impl Into<ArtifactKey>, path: PathBuf) {
        self.inner.insert(key.into(), path);
    }
    pub fn get(&self, key: &str) -> Option<&PathBuf> {
        self.inner.get(key)
    }
    pub fn require(&self, key: &str) -> Result<&PathBuf, String> {
        self.inner
            .get(key)
            .ok_or_else(|| format!("required artifact '{key}' missing"))
    }
    pub fn iter(&self) -> impl Iterator<Item = (&String, &PathBuf)> {
        self.inner.iter()
    }
}
