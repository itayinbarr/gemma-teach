//! Shared path-scoping helpers.
//!
//! Every tool resolves the given relative path against the session's
//! `working_dir` and refuses anything that, after canonicalization, escapes
//! it (including via symlinks). When the resolved path does not yet exist
//! (e.g., for Write), we canonicalize the closest existing ancestor and
//! re-attach the remainder, so we still detect symlink escapes through
//! existing parent directories.

use std::path::{Component, Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum PathError {
    #[error("path '{0}' is empty")]
    Empty(String),
    #[error("absolute paths are not allowed inside a session; use a path relative to the working directory: '{0}'")]
    Absolute(String),
    #[error("path '{0}' escapes the session's working directory")]
    Escapes(String),
    #[error("could not canonicalize working directory '{0}': {1}")]
    CanonWd(String, String),
}

/// Resolve `requested` against `working_dir` and check it is inside the
/// canonical working_dir. Tolerates targets that don't yet exist.
pub fn resolve_scoped(working_dir: &Path, requested: &str) -> Result<PathBuf, PathError> {
    if requested.is_empty() {
        return Err(PathError::Empty(requested.into()));
    }
    let p = Path::new(requested);
    if p.is_absolute() {
        return Err(PathError::Absolute(requested.into()));
    }
    // Reject any explicit `..` traversal up-front; canonicalize() would also
    // catch it, but this gives a clearer error message early.
    for c in p.components() {
        if matches!(c, Component::ParentDir) {
            return Err(PathError::Escapes(requested.into()));
        }
    }
    let wd = working_dir
        .canonicalize()
        .map_err(|e| PathError::CanonWd(working_dir.display().to_string(), e.to_string()))?;
    let joined = wd.join(p);

    // Walk ancestors of `joined` to find the deepest existing one, canonicalize that,
    // then re-attach the remainder. Detects symlink-jumps through real ancestors.
    let (existing, rest) = split_existing(&joined);
    let canon_existing = existing.canonicalize().map_err(|e| {
        PathError::CanonWd(existing.display().to_string(), e.to_string())
    })?;
    let final_path = if rest.as_os_str().is_empty() {
        canon_existing
    } else {
        canon_existing.join(rest)
    };

    if !final_path.starts_with(&wd) {
        return Err(PathError::Escapes(requested.into()));
    }
    Ok(final_path)
}

fn split_existing(p: &Path) -> (PathBuf, PathBuf) {
    if p.exists() {
        return (p.to_path_buf(), PathBuf::new());
    }
    let mut existing = p.to_path_buf();
    let mut suffix: Vec<std::ffi::OsString> = Vec::new();
    loop {
        let parent_owned = existing.parent().map(|p| p.to_path_buf());
        let name_owned = existing.file_name().map(|n| n.to_os_string());
        match (parent_owned, name_owned) {
            (Some(parent), Some(name)) => {
                suffix.push(name);
                existing = parent;
                if existing.exists() {
                    let rest = suffix.iter().rev().fold(PathBuf::new(), |mut acc, n| {
                        acc.push(n);
                        acc
                    });
                    return (existing, rest);
                }
            }
            _ => return (existing, PathBuf::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn rejects_absolute() {
        let wd = tempdir().unwrap();
        let err = resolve_scoped(wd.path(), "/etc/passwd").unwrap_err();
        assert!(matches!(err, PathError::Absolute(_)));
    }

    #[test]
    fn rejects_parent_dir_traversal() {
        let wd = tempdir().unwrap();
        let err = resolve_scoped(wd.path(), "../escape.txt").unwrap_err();
        assert!(matches!(err, PathError::Escapes(_)));
    }

    #[test]
    fn allows_nested_relative_paths_even_if_missing() {
        let wd = tempdir().unwrap();
        let p = resolve_scoped(wd.path(), "a/b/c.md").unwrap();
        assert!(p.starts_with(wd.path().canonicalize().unwrap()));
        assert!(p.ends_with("a/b/c.md"));
    }
}
