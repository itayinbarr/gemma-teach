//! Deterministic output parser.
//!
//! Small local models routinely emit tool calls that *look* right but aren't
//! valid JSON, or are not tool calls at all but prose that *implies* a tool
//! call. The harness's job is to recover from these patterns — we strengthen
//! the parser over time by adding rules that capture what we've seen real
//! models emit, rather than fighting the model with prompts alone.
//!
//! Repair pipeline:
//!
//! 1. Strip `<think>...</think>` blocks (kept aside; surfaced separately).
//! 2. Extract tool-call payloads from `<tool_call>...</tool_call>` XML tags.
//! 3. Extract tool-call payloads from JSON-shaped fenced ```tool/```json blocks.
//! 4. **Gemma quirk passes**: handle `tool_code` fences with Python/prose
//!    syntax, and bare "Write <path>\n```<lang>\n<body>\n```" patterns where
//!    the model dumps a file body in a markdown fence after naming a tool.
//! 5. JSON repair on individual payloads — trailing commas, unquoted keys,
//!    single quotes, literal-newline escaping, brace balancing.
//! 6. `_raw` sentinel for unrepairable payloads so the quality monitor sees
//!    the malformation downstream.

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
    /// Found inside a ``` fence with a JSON payload.
    Fenced,
    /// Found inside `<tool_call>...</tool_call>` XML tags.
    Xml,
    /// Bare JSON object embedded somewhere in the text.
    Bare,
    /// Reconstructed from a Gemma-style `tool_code` fence.
    GemmaToolCode,
    /// Reconstructed from a "Write <path>\n```<lang>\n<body>\n```" pattern.
    GemmaProse,
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
static XML_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>").expect("XML_RE")
});
static UNQUOTED_KEY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:"#).expect("UNQUOTED_KEY_RE"));
static TRAILING_COMMA_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r",\s*([\}\]])").expect("TRAILING_COMMA_RE"));

/// Match any ```<lang>?\n…\n``` fence. We don't use this directly anymore —
/// `extract_fences` walks the input char-by-char to support multiple, possibly
/// adjacent, fences and to preserve byte offsets.
fn extract_fences(s: &str) -> Vec<Fence> {
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i + 3 <= bytes.len() {
        if &bytes[i..i + 3] == b"```" {
            let open_start = i;
            // language tag = until newline
            let mut j = i + 3;
            while j < bytes.len() && bytes[j] != b'\n' {
                j += 1;
            }
            let lang_end = j;
            let lang = std::str::from_utf8(&bytes[i + 3..lang_end]).unwrap_or("").trim().to_string();
            let body_start = (j + 1).min(bytes.len());
            // find closing ```
            let mut k = body_start;
            while k + 3 <= bytes.len() {
                if &bytes[k..k + 3] == b"```" {
                    break;
                }
                k += 1;
            }
            if k + 3 > bytes.len() {
                // Unclosed fence — stop scanning.
                break;
            }
            let body_end = k;
            let close_end = k + 3;
            let body = std::str::from_utf8(&bytes[body_start..body_end])
                .unwrap_or("")
                .trim_end_matches('\n')
                .to_string();
            out.push(Fence {
                outer: open_start..close_end,
                lang,
                body,
            });
            i = close_end;
        } else {
            i += 1;
        }
    }
    out
}

#[derive(Debug, Clone)]
struct Fence {
    outer: std::ops::Range<usize>,
    lang: String,
    body: String,
}

/// Public entry point.
pub fn parse_assistant_output(raw: &str, allow_bare_json: bool) -> ParseOutcome {
    let mut steer = Vec::new();
    let mut tool_calls: Vec<ParsedToolCall> = Vec::new();

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
    for (range, _) in xml_spans.iter().rev() {
        working.replace_range(range.clone(), "");
    }

    // 3-4. Walk fenced blocks. Each fence gets classified:
    //   - `tool_code` / `python_tag` / `python_tool_call`  → Gemma's native tool fence.
    //   - JSON-looking content with name+args              → conventional fenced JSON.
    //   - markdown/text content preceded by "Write <path>" → Gemma prose pattern.
    let fences = extract_fences(&working);
    let mut consumed: Vec<std::ops::Range<usize>> = Vec::new();
    let mut any_quirk = false;
    let mut any_fenced_json = false;

    for (i, fence) in fences.iter().enumerate() {
        // Skip if a previous pass already consumed this byte range.
        if consumed.iter().any(|r| ranges_overlap(r, &fence.outer)) {
            continue;
        }
        let lang_norm = fence.lang.to_ascii_lowercase();

        if is_tool_code_lang(&lang_norm) {
            if let Some(call) = parse_gemma_tool_code(&fence.body) {
                tool_calls.push(ParsedToolCall {
                    call,
                    source: CallSource::GemmaToolCode,
                    had_repair: true,
                    unrepairable: false,
                });
                consumed.push(fence.outer.clone());
                any_quirk = true;
                continue;
            }
        }

        if looks_like_tool_call_payload(&fence.body) {
            if let Some(call) = parse_one_call_payload(&fence.body, CallSource::Fenced) {
                tool_calls.push(call);
                consumed.push(fence.outer.clone());
                any_fenced_json = true;
                continue;
            }
        }

        // Gemma "Write <path>\n```<lang>\n<body>\n```" prose pattern: look
        // at the text immediately preceding this fence. If it ends with
        // `Write <path>` (or Create / Save), synthesize a Write call.
        let preceding = &working[..fence.outer.start];
        if let Some((verb, path)) = sniff_verb_path_prefix(preceding) {
            if let Some(call) = synthesize_write_from_prose(&verb, &path, &fence.body) {
                tool_calls.push(ParsedToolCall {
                    call,
                    source: CallSource::GemmaProse,
                    had_repair: true,
                    unrepairable: false,
                });
                // Consume the verb-and-path line as well as the fence so
                // they don't leak into the visible `text`.
                let line_start = preceding
                    .rfind(verb_anchor(&verb).as_str())
                    .unwrap_or(fence.outer.start);
                consumed.push(line_start..fence.outer.end);
                any_quirk = true;
                continue;
            }
        }

        let _ = i;
    }

    if any_quirk || any_fenced_json {
        steer.push(SteerReason::EmbeddedToolCall);
    }

    // Remove consumed ranges from the working text, largest start-offset first.
    consumed.sort_by(|a, b| b.start.cmp(&a.start));
    for range in &consumed {
        if range.end <= working.len() && range.start <= range.end {
            working.replace_range(range.clone(), "");
        }
    }

    // 5. Bare JSON tool calls (optional). Only top-level `{...}` objects that
    // look like a call.
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

fn ranges_overlap(a: &std::ops::Range<usize>, b: &std::ops::Range<usize>) -> bool {
    a.start < b.end && b.start < a.end
}

fn collapse_whitespace(s: &str) -> String {
    s.trim().to_string()
}

fn is_tool_code_lang(lang: &str) -> bool {
    matches!(
        lang,
        "tool_code" | "tool" | "python_tag" | "python_tool_call" | "function_call" | "tool_call" | "tools"
    )
}

fn looks_like_tool_call_payload(s: &str) -> bool {
    let s = s.trim();
    if !s.starts_with('{') {
        return false;
    }
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

// -----------------------------------------------------------------------------
// Gemma quirk parsers.
//
// Pattern A — `tool_code` fence:
//
//   ```tool_code
//   Read student.md
//   ```
//
// or
//
//   ```tool_code
//   Write(path="student.md", content="# Maya\n…")
//   ```
//
// Pattern B — "Write <path>\n```<lang>\n<body>\n```" prose dump:
//
//   Write student.md
//   ```markdown
//   # Maya
//   …
//   ```
//   Done.
// -----------------------------------------------------------------------------

static TOOL_VERBS: &[&str] = &["Read", "Write", "Edit", "Create", "Save", "View", "Open"];

fn parse_gemma_tool_code(body: &str) -> Option<RawToolCall> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Try Python-style call: `Name(args)`.
    if let Some(call) = parse_python_call(trimmed) {
        return Some(call);
    }

    // Plain `<Verb> <arg>` prose. Pick the first line and parse it.
    let first_line = trimmed.lines().next().unwrap_or(trimmed).trim();
    if let Some((verb, rest)) = first_line.split_once(char::is_whitespace) {
        let cap = capitalize(verb.trim().trim_end_matches([':', ',', '.']));
        if TOOL_VERBS.contains(&cap.as_str()) {
            return Some(build_prose_call(&cap, rest.trim()));
        }
    }
    // Or: a bare verb on a line by itself with nothing after — rare, but
    // emit it with empty path; quality monitor will catch it.
    let single = first_line.trim_end_matches([':', ',', '.']).trim();
    let cap = capitalize(single);
    if TOOL_VERBS.contains(&cap.as_str()) {
        return Some(build_prose_call(&cap, ""));
    }
    None
}

fn parse_python_call(s: &str) -> Option<RawToolCall> {
    // `Name(<args>)` — match opening `(` immediately after an identifier.
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let name_start = i;
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
        i += 1;
    }
    let name = std::str::from_utf8(&bytes[name_start..i]).ok()?.to_string();
    if name.is_empty() {
        return None;
    }
    let cap = capitalize(&name);
    if !TOOL_VERBS.contains(&cap.as_str()) {
        return None;
    }
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'(' {
        return None;
    }
    i += 1;
    // Find matching `)`. Track string literals so `(` inside strings is ignored.
    let arg_start = i;
    let mut depth = 1i32;
    let mut in_str = false;
    let mut quote = b'"';
    let mut esc = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == quote {
                in_str = false;
            }
        } else {
            match b {
                b'"' | b'\'' => {
                    in_str = true;
                    quote = b;
                }
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    if depth != 0 {
        return None;
    }
    let arg_str = std::str::from_utf8(&bytes[arg_start..i]).ok()?;
    let args_value = parse_python_kwargs(arg_str);
    Some(RawToolCall {
        id: ToolCallId::new(),
        name: cap,
        args: args_value,
    })
}

/// Parse a Python-style argument list. Supports keyword args (`k="v"`) and a
/// single positional arg (mapped to a tool-appropriate default key).
fn parse_python_kwargs(s: &str) -> serde_json::Value {
    let s = s.trim();
    if s.is_empty() {
        return serde_json::json!({});
    }
    // First, try to detect keyword args by finding an `=` outside strings.
    let mut obj = serde_json::Map::new();
    let mut buf = Vec::new();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut quote = b'"';
    let mut esc = false;
    for b in s.bytes() {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == quote {
                in_str = false;
            }
            buf.push(b);
            continue;
        }
        match b {
            b'"' | b'\'' => {
                in_str = true;
                quote = b;
                buf.push(b);
            }
            b'(' | b'[' | b'{' => {
                depth += 1;
                buf.push(b);
            }
            b')' | b']' | b'}' => {
                depth -= 1;
                buf.push(b);
            }
            b',' if depth == 0 => {
                if let Some((k, v)) = split_kwarg(std::str::from_utf8(&buf).unwrap_or("")) {
                    obj.insert(k, v);
                }
                buf.clear();
            }
            _ => buf.push(b),
        }
    }
    if !buf.is_empty() {
        if let Some((k, v)) = split_kwarg(std::str::from_utf8(&buf).unwrap_or("")) {
            obj.insert(k, v);
        } else if obj.is_empty() {
            // Single positional argument — map to "path" by default since
            // most tool verbs we care about take a path.
            let single = std::str::from_utf8(&buf).unwrap_or("").trim();
            if !single.is_empty() {
                let v = unquote(single);
                obj.insert("path".into(), serde_json::Value::String(v));
            }
        }
    }
    serde_json::Value::Object(obj)
}

fn split_kwarg(s: &str) -> Option<(String, serde_json::Value)> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Find the first `=` outside strings.
    let bytes = trimmed.as_bytes();
    let mut in_str = false;
    let mut quote = b'"';
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == quote {
                in_str = false;
            }
        } else {
            match b {
                b'"' | b'\'' => {
                    in_str = true;
                    quote = b;
                }
                b'=' => {
                    let key = trimmed[..i].trim().to_string();
                    let raw_val = trimmed[i + 1..].trim();
                    let v = unquote(raw_val);
                    return Some((key, serde_json::Value::String(v)));
                }
                _ => {}
            }
        }
    }
    None
}

fn unquote(s: &str) -> String {
    let t = s.trim();
    if (t.starts_with('"') && t.ends_with('"') && t.len() >= 2)
        || (t.starts_with('\'') && t.ends_with('\'') && t.len() >= 2)
    {
        // strip outer quotes, decode common escapes
        let inner = &t[1..t.len() - 1];
        let mut out = String::with_capacity(inner.len());
        let mut chars = inner.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('r') => out.push('\r'),
                    Some('"') => out.push('"'),
                    Some('\'') => out.push('\''),
                    Some('\\') => out.push('\\'),
                    Some(other) => {
                        out.push('\\');
                        out.push(other);
                    }
                    None => break,
                }
            } else {
                out.push(c);
            }
        }
        out
    } else {
        t.to_string()
    }
}

fn capitalize(s: &str) -> String {
    let s = s.trim();
    if s.is_empty() {
        return String::new();
    }
    // Match against the known tool verbs case-insensitively, return the
    // canonical capitalization.
    for v in TOOL_VERBS {
        if v.eq_ignore_ascii_case(s) {
            return (*v).to_string();
        }
    }
    // Otherwise, capitalize the first letter.
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_ascii_uppercase().to_string() + chars.as_str(),
        None => String::new(),
    }
}

fn build_prose_call(verb: &str, rest: &str) -> RawToolCall {
    // Map verb → tool name and the single argument → the obvious key.
    let (name, key) = match verb {
        "Read" | "Open" | "View" => ("Read", "path"),
        "Write" | "Create" | "Save" => ("Write", "path"),
        "Edit" => ("Edit", "path"),
        _ => (verb, "path"),
    };
    let arg_value = strip_trailing_punctuation(rest);
    let args = serde_json::json!({ key: arg_value });
    RawToolCall {
        id: ToolCallId::new(),
        name: name.to_string(),
        args,
    }
}

fn strip_trailing_punctuation(s: &str) -> String {
    let s = s.trim();
    s.trim_end_matches([',', '.', ';', ':', '!'])
        .trim_matches('`')
        .trim_matches('\'')
        .trim_matches('"')
        .to_string()
}

/// Search backward for "Write <path>" / "Create <path>" / "Save <path>" near
/// the end of the preceding text. Used by the GemmaProse path.
fn sniff_verb_path_prefix(preceding: &str) -> Option<(String, String)> {
    // Take the last non-empty line of `preceding` (strip a trailing newline
    // first so we don't grab the empty string between two newlines).
    let trimmed = preceding.trim_end_matches(['\n', '\r']);
    let tail = trimmed.lines().last().unwrap_or("").trim();
    if tail.is_empty() {
        return None;
    }
    let mut parts = tail.split_whitespace();
    let verb_raw = parts.next()?;
    let path_raw = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    let cap = capitalize(verb_raw.trim_end_matches([':', ',', '.']));
    if !matches!(cap.as_str(), "Write" | "Create" | "Save" | "Edit") {
        return None;
    }
    let path = strip_trailing_punctuation(path_raw);
    if path.is_empty() {
        return None;
    }
    Some((cap, path))
}

fn verb_anchor(verb: &str) -> String {
    verb.to_string()
}

fn synthesize_write_from_prose(verb: &str, path: &str, body: &str) -> Option<RawToolCall> {
    let (name, content_key) = match verb {
        "Write" | "Create" | "Save" => ("Write", "content"),
        "Edit" => {
            // Edit needs old_text + new_text; the prose pattern doesn't carry
            // old_text so we don't synthesize an Edit from it. The quality
            // monitor will surface the missing tool call.
            return None;
        }
        _ => return None,
    };
    let args = serde_json::json!({
        "path": path,
        content_key: body,
    });
    Some(RawToolCall {
        id: ToolCallId::new(),
        name: name.to_string(),
        args,
    })
}

/// Try direct parse, then iteratively apply repair passes. Returns
/// (value, had_repair, unrepairable).
pub fn repair_and_parse(raw: &str) -> (serde_json::Value, bool, bool) {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
        return (v, false, false);
    }

    let mut current = raw.to_string();
    let mut any_repair = false;

    let fixed = escape_unescaped_newlines_in_strings(&current);
    if fixed != current {
        any_repair = true;
        current = fixed;
        if let Ok(v) = serde_json::from_str(&current) {
            return (v, true, false);
        }
    }

    let fixed = TRAILING_COMMA_RE.replace_all(&current, "$1").to_string();
    if fixed != current {
        any_repair = true;
        current = fixed;
        if let Ok(v) = serde_json::from_str(&current) {
            return (v, true, false);
        }
    }

    if !current.contains('"') && current.contains('\'') {
        current = current.replace('\'', "\"");
        any_repair = true;
        if let Ok(v) = serde_json::from_str(&current) {
            return (v, true, false);
        }
    }

    let fixed = quote_unquoted_keys(&current);
    if fixed != current {
        any_repair = true;
        current = fixed;
        if let Ok(v) = serde_json::from_str(&current) {
            return (v, true, false);
        }
    }

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
    }

    #[test]
    fn unquoted_keys_repaired() {
        let s = "<tool_call>{name: \"Read\", args: {path: \"x.md\"}}</tool_call>";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].call.name, "Read");
        assert_eq!(p.tool_calls[0].call.args["path"], "x.md");
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
        let s = "```tool\n{\"name\":\"Write\",\"args\":{\"path\":\"a.md\",\"content\":\"line1\nline2\"}}\n```";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].call.args["content"], "line1\nline2");
    }

    #[test]
    fn missing_closing_brace_repaired() {
        let s = "<tool_call>{\"name\":\"Read\",\"args\":{\"path\":\"x.md\"}</tool_call>";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].call.name, "Read");
    }

    #[test]
    fn unrepairable_emits_raw_sentinel() {
        let s = "<tool_call>not json at all</tool_call>";
        let p = parse(s);
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
        let s = "```python\nprint('hi')\n```";
        let p = parse(s);
        assert!(p.tool_calls.is_empty());
    }

    // ---------- Gemma-quirk fixtures ----------

    /// Captured from Gemma 3n E2B (trace 2026-05-15 student-add 001, extract-tags).
    #[test]
    fn gemma_quirk_tool_code_prose_read() {
        let s = "```tool_code\nRead student.md\n```";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 1, "got: {:?}", p);
        assert_eq!(p.tool_calls[0].call.name, "Read");
        assert_eq!(p.tool_calls[0].call.args["path"], "student.md");
        assert!(matches!(p.tool_calls[0].source, CallSource::GemmaToolCode));
    }

    #[test]
    fn gemma_quirk_tool_code_python_call() {
        let s = "```tool_code\nWrite(path=\"student.md\", content=\"# Maya\\n\\nhello\")\n```";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].call.name, "Write");
        assert_eq!(p.tool_calls[0].call.args["path"], "student.md");
        assert_eq!(p.tool_calls[0].call.args["content"], "# Maya\n\nhello");
    }

    #[test]
    fn gemma_quirk_tool_code_python_positional() {
        let s = "```tool_code\nRead(\"student.md\")\n```";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].call.name, "Read");
        assert_eq!(p.tool_calls[0].call.args["path"], "student.md");
    }

    /// Captured from Gemma 3n E2B (trace 2026-05-15 student-add 001, write-student).
    #[test]
    fn gemma_quirk_prose_write_then_fenced_body() {
        let s = "Write student.md\n```markdown\n# Maya\n\n## Snapshot\n- 12 years old\n```\nDone.";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 1, "got: {:?}", p);
        let c = &p.tool_calls[0];
        assert_eq!(c.call.name, "Write");
        assert_eq!(c.call.args["path"], "student.md");
        assert!(matches!(c.source, CallSource::GemmaProse));
        let content = c.call.args["content"].as_str().unwrap();
        assert!(content.starts_with("# Maya"));
        assert!(content.contains("12 years old"));
    }

    #[test]
    fn gemma_quirk_prose_create_then_fenced_body() {
        let s = "Create notes.md\n```\nshort body\n```";
        let p = parse(s);
        assert_eq!(p.tool_calls.len(), 1);
        assert_eq!(p.tool_calls[0].call.name, "Write");
        assert_eq!(p.tool_calls[0].call.args["path"], "notes.md");
    }

    #[test]
    fn gemma_quirk_does_not_misfire_on_random_text_plus_fence() {
        // No verb+path prefix → should NOT be turned into a Write call.
        let s = "Here is some commentary about the topic.\n```python\nprint('x')\n```";
        let p = parse(s);
        assert!(p.tool_calls.is_empty(), "got: {:?}", p);
    }
}
