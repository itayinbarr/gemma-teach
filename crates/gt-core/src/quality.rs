//! Quality monitor.
//!
//! Inspects each completed turn (parsed assistant output) and decides whether
//! it represents a structural failure that the harness should *correct in
//! place* before another turn is wasted. Detects five failure classes; each
//! maps to a **prescriptive** correction message that includes the literal
//! JSON shape of the desired next call.
//!
//! Corrections are delivered as **steers** — the session loop injects them
//! into the next backend call's message buffer in the same loop iteration, not
//! as a new appended turn. This matches the little-coder pattern.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::parser::ParseOutcome;
use crate::session_event::{CorrectionAction, QualityIssueKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    pub kind: QualityIssueKind,
    pub action: CorrectionAction,
}

/// Snapshot of recent assistant tool-call activity in the current session.
/// We compare each new call against the previous turn's calls (exact name +
/// args JSON equality) to detect loops.
#[derive(Debug, Clone, Default)]
pub struct RecentCalls {
    pub previous: Vec<(String, serde_json::Value)>, // (name, args) from the previous turn
}

#[derive(Debug, Default)]
pub struct QualityMonitor {
    consecutive_corrections: u32,
    max_consecutive: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum CorrectionVerdict {
    Ok,
    Correctable,
    Aborting, // exceeded the consecutive-correction cap
}

impl QualityMonitor {
    pub fn new(max_consecutive: u32) -> Self {
        Self {
            consecutive_corrections: 0,
            max_consecutive,
        }
    }

    pub fn reset_streak(&mut self) {
        self.consecutive_corrections = 0;
    }

    pub fn consecutive(&self) -> u32 {
        self.consecutive_corrections
    }

    /// Run all checks against the turn's `ParseOutcome` and the previous turn's
    /// tool calls. Returns a list of issues (may be empty) along with whether
    /// we should abort due to repeated correction failures.
    pub fn inspect(
        &mut self,
        outcome: &ParseOutcome,
        known_tools: &HashSet<String>,
        recent: &RecentCalls,
    ) -> (Vec<QualityIssue>, CorrectionVerdict) {
        let mut issues = Vec::new();

        // 1. Empty response — no text and no tool calls.
        if outcome.is_empty() {
            issues.push(empty_response());
        }

        // 1b. Premature completion — model emitted only a reply marker
        // (Done.) without making a tool call, when tools were required and
        // no prior tool call has succeeded yet. Distinguishes Gemma 4's
        // shortcut-to-commit failure from legitimate post-tool "Done.".
        if outcome.tool_calls.is_empty()
            && !known_tools.is_empty()
            && recent.previous.is_empty()
            && is_reply_marker_only(&outcome.text)
        {
            issues.push(premature_completion(known_tools));
        }

        // 2-5. Per-call checks.
        for parsed in &outcome.tool_calls {
            // 5. Malformed args (parser couldn't repair to a real object).
            if parsed.unrepairable {
                issues.push(malformed_args(&parsed.call.name));
                continue;
            }
            // 2. Empty tool name.
            if parsed.call.name.trim().is_empty() {
                issues.push(empty_tool_name(known_tools));
                continue;
            }
            // 3. Hallucinated tool name.
            if !known_tools.is_empty() && !known_tools.contains(&parsed.call.name) {
                issues.push(hallucinated_tool(&parsed.call.name, known_tools));
                continue;
            }
            // 4. Repeated tool call (exact match with a previous-turn call).
            for (pname, pargs) in &recent.previous {
                if pname == &parsed.call.name && pargs == &parsed.call.args {
                    issues.push(repeated_tool_call(&parsed.call.name));
                    break;
                }
            }
        }

        if issues.is_empty() {
            self.consecutive_corrections = 0;
            return (issues, CorrectionVerdict::Ok);
        }

        self.consecutive_corrections += 1;
        let verdict = if self.consecutive_corrections > self.max_consecutive {
            CorrectionVerdict::Aborting
        } else {
            CorrectionVerdict::Correctable
        };
        (issues, verdict)
    }
}

// -----------------------------------------------------------------------------
// Correction message builders. Each is **prescriptive** — it tells the model
// the exact shape of the next call, not vague guidance.
// -----------------------------------------------------------------------------

/// True if the text is only a commit-phrase like "Done.", "done", "ok", etc.
/// — i.e., a reply marker the prompts ask for AFTER a tool call succeeds.
fn is_reply_marker_only(text: &str) -> bool {
    let t = text.trim().trim_end_matches(['.', '!']).trim().to_ascii_lowercase();
    matches!(t.as_str(), "done" | "ok" | "okay" | "finished" | "complete" | "")
        && !text.trim().is_empty()
}

fn premature_completion(known: &HashSet<String>) -> QualityIssue {
    let available = format_known(known);
    QualityIssue {
        kind: QualityIssueKind::PrematureCompletion,
        action: CorrectionAction {
            message: format!(
                "You replied with a commit marker (e.g. \"Done.\") before making any tool call. \
                The task is NOT complete until the required file exists. Call the tool first, \
                then reply Done. Available tools: {available}. Example shape:\n\
                ```tool_code\nWrite(path=\"<filename>\", content=\"<file body>\")\n```"
            ),
        },
    }
}

fn empty_response() -> QualityIssue {
    QualityIssue {
        kind: QualityIssueKind::EmptyResponse,
        action: CorrectionAction {
            message: "Your previous response was empty. Respond with either plain text or a native tool call. If the task is fully done, reply exactly: Done.".into(),
        },
    }
}

fn empty_tool_name(known: &HashSet<String>) -> QualityIssue {
    let available = format_known(known);
    QualityIssue {
        kind: QualityIssueKind::EmptyToolName,
        action: CorrectionAction {
            message: format!(
                "Your tool call had an empty `name`. Specify a valid tool name. Available tools: {}. Example shape:\n{{\"name\":\"Read\",\"args\":{{\"path\":\"<relative path>\"}}}}",
                available
            ),
        },
    }
}

fn hallucinated_tool(name: &str, known: &HashSet<String>) -> QualityIssue {
    let available = format_known(known);
    QualityIssue {
        kind: QualityIssueKind::HallucinatedTool { name: name.to_string() },
        action: CorrectionAction {
            message: format!(
                "Tool '{name}' does not exist in this session. Available tools: {available}. Re-issue the call using one of them. Example shape:\n{{\"name\":\"Read\",\"args\":{{\"path\":\"<relative path>\"}}}}"
            ),
        },
    }
}

fn repeated_tool_call(name: &str) -> QualityIssue {
    QualityIssue {
        kind: QualityIssueKind::RepeatedToolCall { tool: name.to_string() },
        action: CorrectionAction {
            message: format!(
                "You just made the same `{name}` call with the same arguments. This suggests you are stuck in a loop. Either change the arguments, switch to a different tool, or — if the task is complete — reply: Done."
            ),
        },
    }
}

fn malformed_args(name: &str) -> QualityIssue {
    let nm = if name.is_empty() { "?" } else { name };
    QualityIssue {
        kind: QualityIssueKind::MalformedArgs { tool: name.to_string() },
        action: CorrectionAction {
            message: format!(
                "Arguments for `{nm}` were not valid JSON. Re-issue as a native tool call with valid JSON. Example shape:\n{{\"name\":\"{nm}\",\"args\":{{\"path\":\"<relative path>\"}}}}"
            ),
        },
    }
}

fn format_known(known: &HashSet<String>) -> String {
    if known.is_empty() {
        return "(none registered)".to_string();
    }
    let mut v: Vec<&String> = known.iter().collect();
    v.sort();
    v.into_iter().cloned().collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{RawToolCall, ToolCallId};
    use crate::parser::{CallSource, ParsedToolCall};

    fn known(names: &[&str]) -> HashSet<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn call(name: &str, args: serde_json::Value) -> ParsedToolCall {
        ParsedToolCall {
            call: RawToolCall {
                id: ToolCallId::new(),
                name: name.into(),
                args,
            },
            source: CallSource::Native,
            had_repair: false,
            unrepairable: false,
        }
    }

    #[test]
    fn empty_response_detected() {
        let mut q = QualityMonitor::new(2);
        let out = ParseOutcome::default();
        let (issues, verdict) = q.inspect(&out, &known(&["Read"]), &RecentCalls::default());
        assert_eq!(issues.len(), 1);
        assert!(matches!(issues[0].kind, QualityIssueKind::EmptyResponse));
        assert!(matches!(verdict, CorrectionVerdict::Correctable));
        assert!(issues[0].action.message.contains("Done."));
    }

    #[test]
    fn empty_tool_name_detected() {
        let mut q = QualityMonitor::new(2);
        let out = ParseOutcome {
            text: "ok".into(),
            thinking: String::new(),
            tool_calls: vec![call("", serde_json::json!({}))],
            steer_reasons: vec![],
        };
        let (issues, _) = q.inspect(&out, &known(&["Read", "Write"]), &RecentCalls::default());
        assert_eq!(issues.len(), 1);
        assert!(matches!(issues[0].kind, QualityIssueKind::EmptyToolName));
        assert!(issues[0].action.message.contains("Read"));
        assert!(issues[0].action.message.contains("Write"));
    }

    #[test]
    fn hallucinated_tool_detected() {
        let mut q = QualityMonitor::new(2);
        let out = ParseOutcome {
            text: String::new(),
            thinking: String::new(),
            tool_calls: vec![call("Frobnicate", serde_json::json!({}))],
            steer_reasons: vec![],
        };
        let (issues, _) = q.inspect(&out, &known(&["Read", "Write"]), &RecentCalls::default());
        assert_eq!(issues.len(), 1);
        match &issues[0].kind {
            QualityIssueKind::HallucinatedTool { name } => assert_eq!(name, "Frobnicate"),
            other => panic!("expected HallucinatedTool, got {:?}", other),
        }
        assert!(issues[0].action.message.contains("Frobnicate"));
    }

    #[test]
    fn repeated_tool_call_detected() {
        let mut q = QualityMonitor::new(2);
        let out = ParseOutcome {
            text: String::new(),
            thinking: String::new(),
            tool_calls: vec![call("Read", serde_json::json!({"path":"a.md"}))],
            steer_reasons: vec![],
        };
        let recent = RecentCalls {
            previous: vec![("Read".into(), serde_json::json!({"path":"a.md"}))],
        };
        let (issues, _) = q.inspect(&out, &known(&["Read"]), &recent);
        assert_eq!(issues.len(), 1);
        assert!(matches!(issues[0].kind, QualityIssueKind::RepeatedToolCall { .. }));
        assert!(issues[0].action.message.contains("loop"));
    }

    #[test]
    fn malformed_args_detected() {
        let mut q = QualityMonitor::new(2);
        let parsed = ParsedToolCall {
            call: RawToolCall {
                id: ToolCallId::new(),
                name: "Read".into(),
                args: serde_json::json!({"_raw":"garbage"}),
            },
            source: CallSource::Native,
            had_repair: true,
            unrepairable: true,
        };
        let out = ParseOutcome {
            text: String::new(),
            thinking: String::new(),
            tool_calls: vec![parsed],
            steer_reasons: vec![],
        };
        let (issues, _) = q.inspect(&out, &known(&["Read"]), &RecentCalls::default());
        assert_eq!(issues.len(), 1);
        assert!(matches!(issues[0].kind, QualityIssueKind::MalformedArgs { .. }));
    }

    #[test]
    fn ok_when_no_issues() {
        let mut q = QualityMonitor::new(2);
        let out = ParseOutcome {
            text: "great".into(),
            thinking: String::new(),
            tool_calls: vec![call("Read", serde_json::json!({"path":"a.md"}))],
            steer_reasons: vec![],
        };
        let (issues, verdict) = q.inspect(&out, &known(&["Read"]), &RecentCalls::default());
        assert!(issues.is_empty());
        assert!(matches!(verdict, CorrectionVerdict::Ok));
    }

    #[test]
    fn streak_aborts_after_cap() {
        let mut q = QualityMonitor::new(2);
        let out = ParseOutcome::default(); // empty response, will trigger
        let _ = q.inspect(&out, &known(&["Read"]), &RecentCalls::default()); // streak=1, Correctable
        let _ = q.inspect(&out, &known(&["Read"]), &RecentCalls::default()); // streak=2, Correctable
        let (_issues, verdict) = q.inspect(&out, &known(&["Read"]), &RecentCalls::default()); // streak=3, Aborting
        assert!(matches!(verdict, CorrectionVerdict::Aborting));
    }

    #[test]
    fn good_turn_resets_streak() {
        let mut q = QualityMonitor::new(2);
        let bad = ParseOutcome::default();
        let _ = q.inspect(&bad, &known(&["Read"]), &RecentCalls::default());
        assert_eq!(q.consecutive(), 1);
        let good = ParseOutcome {
            text: "Looking at the file now.".into(),
            thinking: String::new(),
            tool_calls: vec![call("Read", serde_json::json!({"path":"a.md"}))],
            steer_reasons: vec![],
        };
        let _ = q.inspect(&good, &known(&["Read"]), &RecentCalls::default());
        assert_eq!(q.consecutive(), 0);
    }

    #[test]
    fn premature_completion_detected_on_done_with_no_tool_call() {
        let mut q = QualityMonitor::new(2);
        let out = ParseOutcome {
            text: "Done.".into(),
            thinking: String::new(),
            tool_calls: vec![],
            steer_reasons: vec![],
        };
        let (issues, verdict) = q.inspect(&out, &known(&["Write"]), &RecentCalls::default());
        assert_eq!(issues.len(), 1, "got: {:?}", issues);
        assert!(matches!(issues[0].kind, QualityIssueKind::PrematureCompletion));
        assert!(matches!(verdict, CorrectionVerdict::Correctable));
        assert!(issues[0].action.message.contains("Write"));
        assert!(issues[0].action.message.contains("Call the tool first"));
    }

    #[test]
    fn done_after_successful_call_is_not_premature() {
        // Previous turn had a successful Write call; current turn's "Done."
        // is the legitimate commit marker — must NOT flag.
        let mut q = QualityMonitor::new(2);
        let out = ParseOutcome {
            text: "Done.".into(),
            thinking: String::new(),
            tool_calls: vec![],
            steer_reasons: vec![],
        };
        let recent = RecentCalls {
            previous: vec![("Write".into(), serde_json::json!({"path":"a.md"}))],
        };
        let (issues, verdict) = q.inspect(&out, &known(&["Write"]), &recent);
        assert!(issues.is_empty(), "got: {:?}", issues);
        assert!(matches!(verdict, CorrectionVerdict::Ok));
    }

    #[test]
    fn done_with_no_tools_available_is_not_premature() {
        // No tools registered → "Done." with no calls is fine.
        let mut q = QualityMonitor::new(2);
        let out = ParseOutcome {
            text: "Done.".into(),
            thinking: String::new(),
            tool_calls: vec![],
            steer_reasons: vec![],
        };
        let (issues, _) = q.inspect(&out, &HashSet::new(), &RecentCalls::default());
        assert!(issues.is_empty(), "got: {:?}", issues);
    }

    #[test]
    fn correction_messages_contain_example_shapes() {
        // Lock the prescriptive content. If anyone weakens these strings, this test breaks.
        let mut q = QualityMonitor::new(2);
        let out = ParseOutcome {
            text: String::new(),
            thinking: String::new(),
            tool_calls: vec![call("Doesnt_Exist", serde_json::json!({}))],
            steer_reasons: vec![],
        };
        let (issues, _) = q.inspect(&out, &known(&["Read", "Write"]), &RecentCalls::default());
        let msg = &issues[0].action.message;
        // Must show available tools alphabetized
        assert!(msg.contains("Read, Write"));
        // Must include the literal JSON shape
        assert!(msg.contains("\"name\":\"Read\""));
        assert!(msg.contains("\"args\""));
    }
}
