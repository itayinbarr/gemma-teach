//! Slash-command parsing.

#[derive(Debug, Clone)]
pub enum Slash {
    StudentAdd,
    ClassPlan { pdf: std::path::PathBuf },
    /// `/class-plan` with no argument — open the input modal.
    ClassPlanModal,
    StudentEdit { name: String },
    Help,
    Quit,
    Unknown(String),
}

pub fn parse(line: &str) -> Option<Slash> {
    let trimmed = line.trim();
    let body = trimmed.strip_prefix('/')?;
    let mut parts = body.split_whitespace();
    let head = parts.next()?;
    let rest: Vec<String> = parts.map(|s| s.to_string()).collect();
    match head {
        "student-add" => Some(Slash::StudentAdd),
        "class-plan" => {
            if rest.is_empty() {
                Some(Slash::ClassPlanModal)
            } else {
                let raw = rest.join(" ");
                // Expand a leading `~/` so users can type ~/Downloads/foo.pdf.
                let expanded = if let Some(stripped) = raw.strip_prefix("~/") {
                    dirs::home_dir()
                        .map(|h| h.join(stripped))
                        .unwrap_or_else(|| std::path::PathBuf::from(raw.clone()))
                } else {
                    std::path::PathBuf::from(raw)
                };
                Some(Slash::ClassPlan { pdf: expanded })
            }
        }
        "student-edit" => {
            if rest.is_empty() {
                Some(Slash::Unknown(
                    "/student-edit needs a student slug: /student-edit maya".into(),
                ))
            } else {
                Some(Slash::StudentEdit {
                    name: rest.join(" "),
                })
            }
        }
        "help" | "?" => Some(Slash::Help),
        "quit" | "q" | "exit" => Some(Slash::Quit),
        other => Some(Slash::Unknown(format!("unknown command: /{}", other))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn student_add() {
        assert!(matches!(parse("/student-add"), Some(Slash::StudentAdd)));
        assert!(matches!(parse("  /student-add  "), Some(Slash::StudentAdd)));
    }

    #[test]
    fn class_plan_no_arg_opens_modal() {
        assert!(matches!(parse("/class-plan"), Some(Slash::ClassPlanModal)));
    }

    #[test]
    fn class_plan_with_path() {
        match parse("/class-plan /tmp/foo.pdf").unwrap() {
            Slash::ClassPlan { pdf } => assert_eq!(pdf.to_str(), Some("/tmp/foo.pdf")),
            _ => panic!(),
        }
    }

    #[test]
    fn class_plan_expands_tilde() {
        match parse("/class-plan ~/Documents/x.pdf").unwrap() {
            Slash::ClassPlan { pdf } => {
                // Should not start with `~` after expansion (assuming HOME is set).
                if let Some(home) = dirs::home_dir() {
                    assert!(pdf.starts_with(&home), "got: {}", pdf.display());
                }
            }
            _ => panic!(),
        }
    }

    #[test]
    fn student_edit() {
        match parse("/student-edit maya").unwrap() {
            Slash::StudentEdit { name } => assert_eq!(name, "maya"),
            _ => panic!(),
        }
    }

    #[test]
    fn unknown_prefix() {
        assert!(parse("hello").is_none());
    }
}
