//! Slash-command parsing.

#[derive(Debug, Clone)]
pub enum Slash {
    StudentAdd,
    ClassPlan { pdf: std::path::PathBuf },
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
                Some(Slash::Unknown(
                    "/class-plan needs a path: /class-plan /path/to/chapter.pdf".into(),
                ))
            } else {
                Some(Slash::ClassPlan {
                    pdf: rest.join(" ").into(),
                })
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
    fn class_plan_requires_arg() {
        assert!(matches!(parse("/class-plan"), Some(Slash::Unknown(_))));
        match parse("/class-plan /tmp/foo.pdf").unwrap() {
            Slash::ClassPlan { pdf } => assert_eq!(pdf.to_str(), Some("/tmp/foo.pdf")),
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
