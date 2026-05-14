//! Deterministic output parser.
//!
//! Small local models routinely emit tool calls that *look* right but aren't
//! valid JSON or are wrapped in markdown/XML. Before the harness can interpret
//! anything semantically, we run the model's raw output through a pipeline of
//! repair passes:
//!
//! 1. Strip `<think>...</think>` blocks (kept aside; surfaced separately).
//! 2. Extract tool-call payloads from fenced ```tool / ```json blocks and
//!    `<tool_call>...</tool_call>` tags.
//! 3. For each extracted payload, attempt `serde_json::from_str`; on failure,
//!    apply repair passes (trailing-comma strip, unquoted-key quoting,
//!    single→double quotes, literal-newline escaping inside strings,
//!    brace balancing) and re-parse.
//! 4. If repair still fails, emit a `_raw` sentinel so the quality monitor
//!    can see the malformation downstream.
//!
//! When a fenced/XML wrapping is detected, we also surface an
//! `EmbeddedToolCall` steer reason so the session loop can nudge the model
//! toward native tool-call format on the next turn.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::message::{RawToolCall, ToolCallId};
use crate::session_event::SteerReason;

/// Where this call came from in the model output. Drives the steer reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallSource {
    /// Already a clean native tool call (no parsing needed).
    Native,
    /// Found inside a ``` fence (any language tag).
    Fenced,
    /// Found inside `<tool_call>...</tool_call>` XML tags.
    Xml,
    /// Bare JSON object embedded somewhere in the text.
    Bare,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedToolCall {
    pub call: RawToolCall,
    pub source: CallSource,
    pub had_repair: bool,
    /// If true, this call's args could not be repaired — `call.args` contains
    /// a `{_raw: <original>}` sentinel.
    pub unrepairable: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParseOutcome {
    /// Visible assistant text with `<think>` and tool-call blocks stripped.
    pub text: String,
    /// Concatenated content of any `<think>...</think>` blocks.
    pub thinking: String,
    pub tool_calls: Vec<ParsedToolCall>,
    pub steer_reasons: Vec<SteerReason>,
}

impl ParseOutcome {
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty() && self.tool_calls.is_empty()
    }
}

static THINK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)<think>(.*?)</think>").expect("THINK_RE"));
static FENCED_RE: Lazy<Regex> = Lazy::new(|| {
    // ```<lang>\n...\n```  — capture the inside. Language tag optional.
    Regex::new(r"(?s)```(?:[a-zA-Z0-9_-]*)?\s*\n?(.*?)\n?```").expect("FENCED_RE")
});
static XML_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>").expect("XML_RE")
});
static UNQUOTED_KEY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:"#).expect("UNQUOTED_KEY_RE"));
static TRAILING_COMMA_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r",\s*([\}\]])").expect("TRAILING_COMMA_RE"));

/// Public entry point.
pub fn parse_assistant_output(raw: &str, allow_bare_json: bool) -> ParseOutcome {
    let mut steer = Vec::new();

    // 1. Pull thinking blocks out and accumulate them.
    let mut thinking = String::new();
    let after_think = THINK_RE.replace_all(raw, |c: &regex::Captures| {
        if let Some(m) = c.get(1) {
            if !thinking.is_empty() {
                thinking.push('\n');
            }
            thinking.push_str(m.as_str().trim());
        }
        String::new()
    });

    let mut tool_calls: Vec<ParsedToolCall> = Vec::new();
    let mut working = after_think.into_owned();

    // 2. Extract <tool_call>...</tool_call>.
    let xml_spans: Vec<_> = XML_RE
        .captures_iter(&working)
        .filter_map(|c| {
            let outer = c.get(0)?;
            let inner = c.get(1)?;
            Some((outer.start()..outer.end(), inner.as_str().to_string()))
        })
        .collect();
    if !xml_spans.is_empty() {
        steer.push(SteerReason::EmbeddedToolCall);
    }
    for (_, payload) in &xml_spans {
        if let Some(call) = parse_one_call_payload(payload, CallSource::Xml) {
            tool_calls.push(call);
        }
    }
    // remove them in reverse order so indices stay valid
    for (range, _) in xml_spans.iter().rev() {
        working.replace_range(range.clone(), "");
    }

    // 3. Extract ``` fenced blocks. We only treat a fenced block as a tool
    // call when its content parses to an object that has *either* a "name"
    // or a "tool" key (with companion args / input / arguments). This keeps
    // ordinary code fences in the assistant text from being misread as calls.
    let fenced_spans: Vec<_> = FENCED_RE
        .captures_iter(&working)
        .filter_map(|c| {
            let outer = c.get(0)?;
            let inner = c.get(1)?;
            Some((outer.start()..outer.end(), inner.as_str().to_string()))
        })
        .collect();
    let mut any_fenced_call = false;
    let mut fenced_call_ranges: Vec<std::ops::Range<usize>> = Vec::new();
    for (range, payload) in &fenced_spans {
        if looks_like_tool_call_payload(payload) {
            if let Some(call) = parse_one_call_payload(payload, CallSource::Fenced) {
                tool_calls.push(call);
                any_fenced_call = true;
                fenced_call_ranges.push(range.clone());
            }
        }
    }
    if any_fenced_call {
        steer.push(SteerReason::EmbeddedToolCall);
    }
    for range in fenced_call_ranges.into_iter().rev() {
        working.replace_range(range, "");
    }

    // 4. Bare JSON tool calls (optional — caller decides). We are conservative:
    // only match top-level `{...}` objects that contain a `"name"`/`"tool"`
    // key and look like a call.
    if allow_bare_json {
        let bare = scan_bare_json_calls(&working);
        for (range, payload) in bare.iter().rev() {
            if let Some(call) = parse_one_call_payload(payload, CallSource::Bare) {
                tool_calls.push(call);
                working.replace_range(range.clone(), "");
            }
        }
    }

    ParseOutcome {
        text: collapse_whitespace(&working),
        thinking,
        tool_calls,
        steer_reasons: steer,
    }
}

fn collapse_whitespace(s: &str) -> String {
    s.trim().to_string()
}

fn looks_like_tool_call_payload(s: &str) -> bool {
    let s = s.trim();
    if !s.starts_with('{') {
        return false;
    }
    // cheap heuristic: contains "name" or "tool" key and an args-style sibling
    let has_name = s.contains("\"name\"") || s.contains("name:");
    let has_tool = s.contains("\"tool\"") || s.contains("tool:");
    let has_args = s.contains("\"args\"")
        || s.contains("args:")
        || s.contains("\"input\"")
        || s.contains("input:")
        || s.contains("\"arguments\"")
        || s.contains("arguments:")
        || s.contains("\"parameters\"")
        || s.contains("parameters:");
    (has_name || has_tool) && has_args
}

fn scan_bare_json_calls(s: &str) -> Vec<(std::ops::Range<usize>, String)> {
    // Walk for top-level balanced `{...}` substrings and test each.
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            if let Some(end) = find_matching_brace(bytes, i) {
                let payload = &s[i..=end];
                if looks_like_tool_call_payload(payload) {
                    out.push((i..end + 1, payload.to_string()));
                    i = end + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

fn find_matching_brace(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if *b == b'\\' {
                esc = true;
            } else if *b == b'"' {
                in_str = false;
            }
            continue;
        }
        match *b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_one_call_payload(payload: &str, source: CallSource) -> Option<ParsedToolCall> {
    let trimmed = payload.trim();
    if trimmed.is_empty() {
        return None;
    }

    let (parsed, had_repair, unrepairable) = repair_and_parse(trimmed);
    let obj = match parsed {
        serde_json::Value::Object(m) => m,
        _ => return None,
    };

    // Extract name/tool and args/input/arguments.
    let name = obj
        .get("name")
        .or_else(|| obj.get("tool"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_default();

    let args = obj
        .get("args")
        .cloned()
        .or_else(|| obj.get("input").cloned())
        .or_else(|| obj.get("arguments").cloned())
        .or_else(|| obj.get("parameters").cloned())
        .unwrap_or(serde_json::Value::Object(Default::default()));

    let final_args = if unrepairable {
        serde_json::json!({ "_raw": trimmed })
    } else {
        args
    };

    Some(ParsedToolCall {
        call: RawToolCall {
            id: ToolCallId::new(),
            name,
            args: final_args,
        },
        source,
        had_repair,
        unrepairable,
    })
}

/// Try direct parse, then iteratively apply repair passes. Returns
/// (value, had_repair, unrepairable).
pub fn repair_and_parse(raw: &str) -> (serde_json::Value, bool, bool) {
    // Direct parse
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
        return (v, false, false);
    }

    // Iterative repair.
    let mut current = raw.to_string();
    let mut any_repair = false;

    // 1) escape unescaped literal newlines inside string literals
    let fixed = escape_unescaped_newlines_in_strings(&current);
    if fixed != current {
        any_repair = true;
        current = fixed;
        if let Ok(v) = serde_json::from_str(&current) {
            return (v, true, false);
        }
    }

    // 2) strip trailing commas
    let fixed = TRAILING_COMMA_RE.replace_all(&current, "$1").to_string();
    if fixed != current {
        any_repair = true;
        current = fixed;
        if let Ok(v) = serde_json::from_str(&current) {
            return (v, true, false);
        }
    }

    // 3) single → double quotes, but only if there are no existing double quotes
    if !current.contains('"') && current.contains('\'') {
        current = current.replace('\'', "\"");
        any_repair = true;
        if let Ok(v) = serde_json::from_str(&current) {
            return (v, true, false);
        }
    }

    // 4) quote unquoted keys (only outside string literals)
    let fixed = quote_unquoted_keys(&current);
    if fixed != current {
        any_repair = true;
        current = fixed;
        if let Ok(v) = serde_json::from_str(&current) {
            return (v, true, false);
        }
    }

    // 5) brace balancing — count and append closers
    let open_curly = current.matches('{').count() as i64 - current.matches('}').count() as i64;
    let open_brack = current.matches('[').count() as i64 - current.matches(']').count() as i64;
    if open_curly > 0 || open_brack > 0 {
        for _ in 0..open_brack.max(0) {
            current.push(']');
        }
        for _ in 0..open_curly.max(0) {
            current.push('}');
        }
        any_repair = true;
        if let Ok(v) = serde_json::from_str(&current) {
            return (v, true, false);
        }
    }

    // 6) last resort: extract first `{...}` object substring and try it
    if let Some(start) = current.find('{') {
        if let Some(end) = find_matching_brace(current.as_bytes(), start) {
            let sub = &current[start..=end];
            if let Ok(v) = serde_json::from_str(sub) {
                return (v, true, false);
            }
        }
    }

    (serde_json::json!({ "_raw": raw }), any_repair, true)
}

/// Walk the string, tracking whether we're inside a string literal, and
/// escape any literal `\n` characters that appear inside one.
fn escape_unescaped_newlines_in_strings(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_str = false;
    let mut esc = false;
    for ch in s.chars() {
        if in_str {
            if esc {
                esc = false;
                out.push(ch);
                continue;
            }
            if ch == '\\' {
                esc = true;
                out.push(ch);
                continue;
            }
            if ch == '"' {
                in_str = false;
                out.push(ch);
                continue;
            }
            if ch == '\n' {
                out.push_str("\\n");
                continue;
            }
            if ch == '\r' {
                out.push_str("\\r");
                continue;
            }
            if ch == '\t' {
                out.push_str("\\t");
                continue;
            }
            out.push(ch);
        } else {
            if ch == '"' {
                in_str = true;
            }
            out.push(ch);
        }
    }
    out
}

/// Replace `{ key: ` and `, key: ` patterns with quoted-key versions.
/// We do a regex pass but rely on the fact that we previously escaped any
/// inside-string newlines, so the regex is safe to apply to the whole text.
fn quote_unquoted_keys(s: &str) -> String {
    UNQUOTED_KEY_RE
        .replace_all(s, |c: &regex::Captures| {
            let prefix = c.get(1).map(|m| m.as_str()).unwrap_or("");
            let key = c.get(2).map(|m| m.as_str()).unwrap_or("");
            format!("{}\"{}\":", prefix, key)
        })
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> ParseOutcome {
        parse_assistant_output(s, true)
    }

    #[test]
    fn plain_text_no_calls() {
        let p = parse("just a sentence.");
        assert!(p.tool_calls.is_empty());
        assert_eq!(p.text, "just a sentence.");
        assert!(p.thinking.is_empty());
    }

    #[test]
    fn fenced_tool_call_extracted() {
        let s = "I'll save it.\n```tool\n{\"name\":\"Write\",\"args\":{\"path\":\"a.md\",\"content\":\"hi\"}}\n```";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].call.name, "Write");
        assert!(matches!(p.tool_calls[0].source, CallSource::Fenced));
        assert!(p.steer_reasons.iter().any(|r| matches!(r, SteerReason::EmbeddedToolCall)));
        assert_eq!(p.tool_calls[0].call.args["path"], "a.md");
    }

    #[test]
    fn xml_tool_call_extracted() {
        let s = "<tool_call>{\"name\":\"Read\",\"args\":{\"path\":\"x.md\"}}</tool_call>";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].call.name, "Read");
        assert!(matches!(p.tool_calls[0].source, CallSource::Xml));
    }

    #[test]
    fn trailing_comma_repaired() {
        let s = "<tool_call>{\"name\":\"Read\",\"args\":{\"path\":\"x.md\",},}</tool_call>";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].call.args["path"], "x.md");
        assert!(p.tool_calls[0].had_repair);
        assert!(!p.tool_calls[0].unrepairable);
    }

    #[test]
    fn unquoted_keys_repaired() {
        let s = "<tool_call>{name: \"Read\", args: {path: \"x.md\"}}</tool_call>";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].call.name, "Read");
        assert_eq!(p.tool_calls[0].call.args["path"], "x.md");
        assert!(p.tool_calls[0].had_repair);
    }

    #[test]
    fn single_quotes_repaired() {
        let s = "<tool_call>{'name': 'Read', 'args': {'path': 'x.md'}}</tool_call>";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].call.name, "Read");
    }

    #[test]
    fn literal_newline_in_string_repaired() {
        // raw assistant string contains a literal LF inside the content value
        let s = "```tool\n{\"name\":\"Write\",\"args\":{\"path\":\"a.md\",\"content\":\"line1\nline2\"}}\n```";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].call.args["content"], "line1\nline2");
        assert!(p.tool_calls[0].had_repair);
    }

    #[test]
    fn missing_closing_brace_repaired() {
        let s = "<tool_call>{\"name\":\"Read\",\"args\":{\"path\":\"x.md\"}</tool_call>";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].call.name, "Read");
        assert!(p.tool_calls[0].had_repair);
    }

    #[test]
    fn unrepairable_emits_raw_sentinel() {
        // pathologically broken — empty body
        let s = "<tool_call>not json at all</tool_call>";
        let p = parse(s);
        // unrepairable payloads still emit a call (with empty name and _raw args)
        // so the quality monitor catches them downstream.
        assert_eq!(p.tool_calls.len(), 1);
        assert!(p.tool_calls[0].unrepairable);
        assert!(p.tool_calls[0].call.args.get("_raw").is_some());
    }

    #[test]
    fn thinking_block_stripped() {
        let s = "<think>let me consider...</think>The answer is 42.";
        let p = parse(s);
        assert_eq!(p.thinking, "let me consider...");
        assert_eq!(p.text, "The answer is 42.");
        assert!(p.tool_calls.is_empty());
    }

    #[test]
    fn bare_json_call_extracted_when_allowed() {
        let s = "Sure: {\"name\":\"Read\",\"args\":{\"path\":\"x.md\"}}";
        let p = parse_assistant_output(s, true);
        assert_eq!(p.tool_calls.len(), 1);
        assert!(matches!(p.tool_calls[0].source, CallSource::Bare));
    }

    #[test]
    fn bare_json_ignored_when_disallowed() {
        let s = "Sure: {\"name\":\"Read\",\"args\":{\"path\":\"x.md\"}}";
        let p = parse_assistant_output(s, false);
        assert!(p.tool_calls.is_empty());
    }

    #[test]
    fn alternate_keys_input_and_arguments_supported() {
        let s = "<tool_call>{\"name\":\"Read\",\"input\":{\"path\":\"x.md\"}}</tool_call>";
        let p = parse(s);
        assert_eq!(p.tool_calls[0].call.args["path"], "x.md");

        let s = "<tool_call>{\"tool\":\"Read\",\"arguments\":{\"path\":\"x.md\"}}</tool_call>";
        let p = parse(s);
        assert_eq!(p.tool_calls[0].call.name, "Read");
        assert_eq!(p.tool_calls[0].call.args["path"], "x.md");
    }

    #[test]
    fn multiple_fenced_calls_collected_in_order() {
        let s = "```tool\n{\"name\":\"Read\",\"args\":{\"path\":\"a.md\"}}\n```\nthen\n```tool\n{\"name\":\"Write\",\"args\":{\"path\":\"b.md\",\"content\":\"hi\"}}\n```";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 2);
        assert_eq!(p.tool_calls[0].call.name, "Read");
        assert_eq!(p.tool_calls[1].call.name, "Write");
    }

    #[test]
    fn fenced_code_block_without_call_shape_left_alone() {
        // A code fence that doesn't look like a tool call must not be eaten.
        let s = "```python\nprint('hi')\n```";
        let p = parse(s);
        assert!(p.tool_calls.is_empty());
        // text retains the fence content (we don't try to strip non-call fences here)
    }
}
