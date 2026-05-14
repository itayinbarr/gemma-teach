//! Per-turn skill-card selector.
//!
//! Skill cards live as markdown files with a YAML frontmatter block. Each card
//! targets a specific tool and contains a literal `EXAMPLE` JSON shape for
//! that tool. The injector picks 1–3 cards per turn, ranked by:
//!
//! 1. Error recovery — the last tool whose call failed (+10).
//! 2. Recency — tools used in the last few turns (+3 each).
//! 3. Intent — keyword overlap with `intent_tags` (+2 multi-word, +1 single).
//!
//! Selection is greedy under a token budget (chars/4 heuristic; we don't ship
//! Gemma's tokenizer here — close enough for budgeting).

use serde::Deserialize;
use std::collections::VecDeque;

/// One parsed skill card.
#[derive(Debug, Clone)]
pub struct SkillCard {
    pub name: String,
    pub target_tool: String,
    pub intent_tags: Vec<String>,
    pub token_cost: usize,
    pub body: String,
    pub source_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    name: String,
    target_tool: String,
    #[serde(default)]
    intent_tags: Vec<String>,
    #[serde(default = "default_token_cost")]
    token_cost: usize,
}

fn default_token_cost() -> usize {
    100
}

/// Parse a single skill-card file (YAML frontmatter + body).
pub fn parse_skill_markdown(text: &str, source_path: Option<String>) -> Option<SkillCard> {
    let (front, body) = split_frontmatter(text)?;
    let fm: SkillFrontmatter = serde_yaml::from_str(front).ok()?;
    Some(SkillCard {
        name: fm.name,
        target_tool: fm.target_tool,
        intent_tags: fm.intent_tags,
        token_cost: fm.token_cost,
        body: body.trim().to_string(),
        source_path,
    })
}

pub(crate) fn split_frontmatter(text: &str) -> Option<(&str, &str)> {
    let rest = text.strip_prefix("---\n").or_else(|| text.strip_prefix("---\r\n"))?;
    if let Some(end) = rest.find("\n---") {
        let body_start = end + "\n---".len();
        let after_marker = &rest[body_start..];
        // skip the optional newline directly after the closing marker
        let body = after_marker
            .strip_prefix('\n')
            .or_else(|| after_marker.strip_prefix("\r\n"))
            .unwrap_or(after_marker);
        return Some((&rest[..end], body));
    }
    None
}

/// Tracks recent tool usage for the recency component of skill ranking.
#[derive(Debug, Default, Clone)]
pub struct RecentUsage {
    history: VecDeque<String>,
    capacity: usize,
}

impl RecentUsage {
    pub fn new(capacity: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(capacity),
            capacity,
        }
    }
    pub fn record(&mut self, tool: impl Into<String>) {
        if self.history.len() >= self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(tool.into());
    }
    pub fn iter(&self) -> impl Iterator<Item = &String> {
        self.history.iter()
    }
}

/// Inputs to `SkillInjector::select`.
pub struct SelectionCtx<'a> {
    pub user_prompt: &'a str,
    pub last_failed_tool: Option<&'a str>,
    pub recent: &'a RecentUsage,
    pub token_budget: usize,
    pub allowed_tools: &'a [String],
}

pub struct SkillInjector {
    cards: Vec<SkillCard>,
}

impl SkillInjector {
    pub fn new(cards: Vec<SkillCard>) -> Self {
        Self { cards }
    }

    /// Compute selection scores and pack greedily under the budget.
    pub fn select(&self, ctx: &SelectionCtx<'_>) -> Vec<&SkillCard> {
        let allowed: std::collections::HashSet<&str> =
            ctx.allowed_tools.iter().map(|s| s.as_str()).collect();
        let prompt_lower = ctx.user_prompt.to_lowercase();

        let mut scored: Vec<(i64, &SkillCard)> = self
            .cards
            .iter()
            .filter(|c| allowed.is_empty() || allowed.contains(c.target_tool.as_str()))
            .map(|c| {
                let mut score = 0i64;
                if Some(c.target_tool.as_str()) == ctx.last_failed_tool {
                    score += 10;
                }
                let recent_hits = ctx
                    .recent
                    .iter()
                    .filter(|t| t.as_str() == c.target_tool)
                    .count() as i64;
                score += recent_hits * 3;
                for tag in &c.intent_tags {
                    let tag_lower = tag.to_lowercase();
                    if !tag_lower.contains(' ') {
                        if prompt_lower.split(|ch: char| !ch.is_alphanumeric()).any(|w| w == tag_lower) {
                            score += 1;
                        }
                    } else if prompt_lower.contains(&tag_lower) {
                        score += 2;
                    }
                }
                (score, c)
            })
            .filter(|(s, _)| *s > 0)
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0));

        let mut used = 0usize;
        let mut out = Vec::new();
        for (_, card) in scored {
            if used + card.token_cost > ctx.token_budget && !out.is_empty() {
                break;
            }
            out.push(card);
            used += card.token_cost;
            if used >= ctx.token_budget {
                break;
            }
        }
        out
    }

    pub fn cards(&self) -> &[SkillCard] {
        &self.cards
    }
}

/// Render selected cards as a Markdown block suitable for appending to the
/// system prompt under `## Skills for this turn`.
pub fn render_skills_block(cards: &[&SkillCard]) -> String {
    if cards.is_empty() {
        return String::new();
    }
    let mut s = String::from("## Skills for this turn\n");
    for c in cards {
        s.push('\n');
        s.push_str(&c.body);
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_card(name: &str, tool: &str, tags: &[&str], cost: usize) -> SkillCard {
        SkillCard {
            name: name.into(),
            target_tool: tool.into(),
            intent_tags: tags.iter().map(|s| s.to_string()).collect(),
            token_cost: cost,
            body: format!("## {tool}\nbody"),
            source_path: None,
        }
    }

    #[test]
    fn parse_skill_with_frontmatter() {
        let md = "---\nname: read-file\ntarget_tool: Read\nintent_tags: [read, open]\ntoken_cost: 60\n---\n## Read\nbody text\n";
        let c = parse_skill_markdown(md, None).unwrap();
        assert_eq!(c.name, "read-file");
        assert_eq!(c.target_tool, "Read");
        assert_eq!(c.intent_tags, vec!["read", "open"]);
        assert_eq!(c.token_cost, 60);
        assert!(c.body.contains("body text"));
    }

    #[test]
    fn error_recovery_outranks_recency_and_intent() {
        let cards = vec![
            make_card("write", "Write", &["create"], 50),
            make_card("read", "Read", &["read"], 50),
            make_card("edit", "Edit", &["update"], 50),
        ];
        let injector = SkillInjector::new(cards);
        let mut recent = RecentUsage::new(3);
        recent.record("Edit");
        recent.record("Edit");
        let ctx = SelectionCtx {
            user_prompt: "please read the file",
            last_failed_tool: Some("Write"),
            recent: &recent,
            token_budget: 70, // only one fits
            allowed_tools: &["Read".into(), "Write".into(), "Edit".into()],
        };
        let selected = injector.select(&ctx);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].target_tool, "Write"); // error recovery wins
    }

    #[test]
    fn recency_outranks_intent_when_present() {
        let cards = vec![
            make_card("write", "Write", &["create"], 50),
            make_card("read", "Read", &["read"], 50),
        ];
        let injector = SkillInjector::new(cards);
        let mut recent = RecentUsage::new(3);
        recent.record("Write");
        let ctx = SelectionCtx {
            user_prompt: "please read",
            last_failed_tool: None,
            recent: &recent,
            token_budget: 70,
            allowed_tools: &["Read".into(), "Write".into()],
        };
        let selected = injector.select(&ctx);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].target_tool, "Write"); // recency +3 > intent +1
    }

    #[test]
    fn intent_picks_relevant_when_nothing_else() {
        let cards = vec![
            make_card("write", "Write", &["create", "save", "new file"], 50),
            make_card("read", "Read", &["read", "open"], 50),
            make_card("edit", "Edit", &["update", "fix"], 50),
        ];
        let injector = SkillInjector::new(cards);
        let recent = RecentUsage::new(3);
        let ctx = SelectionCtx {
            user_prompt: "please update the file",
            last_failed_tool: None,
            recent: &recent,
            token_budget: 70,
            allowed_tools: &["Read".into(), "Write".into(), "Edit".into()],
        };
        let selected = injector.select(&ctx);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].target_tool, "Edit");
    }

    #[test]
    fn multi_word_phrase_scores_higher_than_single_word() {
        let cards = vec![
            make_card("a", "Write", &["new file"], 50), // multi-word +2
            make_card("b", "Read", &["new"], 50),       // single +1
        ];
        let injector = SkillInjector::new(cards);
        let recent = RecentUsage::new(3);
        let ctx = SelectionCtx {
            user_prompt: "create a new file please",
            last_failed_tool: None,
            recent: &recent,
            token_budget: 70,
            allowed_tools: &["Read".into(), "Write".into()],
        };
        let selected = injector.select(&ctx);
        assert_eq!(selected[0].target_tool, "Write");
    }

    #[test]
    fn budget_packs_greedily_and_stops() {
        let cards = vec![
            make_card("a", "Write", &["create"], 60),
            make_card("b", "Read", &["read"], 60),
            make_card("c", "Edit", &["update"], 60),
        ];
        let injector = SkillInjector::new(cards);
        let recent = RecentUsage::new(3);
        let ctx = SelectionCtx {
            user_prompt: "create read update",
            last_failed_tool: None,
            recent: &recent,
            token_budget: 130,
            allowed_tools: &["Read".into(), "Write".into(), "Edit".into()],
        };
        let selected = injector.select(&ctx);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn allowed_filter_excludes_disallowed_tools() {
        let cards = vec![
            make_card("write", "Write", &["create"], 50),
            make_card("bash", "Bash", &["run"], 50),
        ];
        let injector = SkillInjector::new(cards);
        let recent = RecentUsage::new(3);
        let ctx = SelectionCtx {
            user_prompt: "create and run",
            last_failed_tool: None,
            recent: &recent,
            token_budget: 200,
            allowed_tools: &["Write".into()],
        };
        let selected = injector.select(&ctx);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].target_tool, "Write");
    }

    #[test]
    fn render_block_returns_empty_for_empty_selection() {
        assert_eq!(render_skills_block(&[]), "");
    }

    #[test]
    fn render_block_includes_card_bodies() {
        let c = make_card("a", "Read", &[], 50);
        let block = render_skills_block(&[&c]);
        assert!(block.contains("## Skills for this turn"));
        assert!(block.contains("body"));
    }
}
