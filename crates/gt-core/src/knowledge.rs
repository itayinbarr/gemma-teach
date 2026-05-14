//! Domain-knowledge cheat-sheet selector.
//!
//! Same shape as `skills` but lower priority and scored on keyword overlap
//! only. Multi-word phrase match = +2.0, single-word = +1.0, threshold 2.0.
//! Greedy pack under a token budget.

use serde::Deserialize;

use crate::skills::split_frontmatter;

#[derive(Debug, Clone)]
pub struct KnowledgeCard {
    pub topic: String,
    pub keywords: Vec<String>,
    pub token_cost: usize,
    pub body: String,
    pub source_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct KnowledgeFrontmatter {
    topic: String,
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default = "default_cost")]
    token_cost: usize,
}

fn default_cost() -> usize {
    100
}

pub fn parse_knowledge_markdown(text: &str, source_path: Option<String>) -> Option<KnowledgeCard> {
    let (front, body) = split_frontmatter(text)?;
    let fm: KnowledgeFrontmatter = serde_yaml::from_str(front).ok()?;
    Some(KnowledgeCard {
        topic: fm.topic,
        keywords: fm.keywords,
        token_cost: fm.token_cost,
        body: body.trim().to_string(),
        source_path,
    })
}

pub struct KnowledgeInjector {
    cards: Vec<KnowledgeCard>,
    threshold: f32,
}

impl KnowledgeInjector {
    pub fn new(cards: Vec<KnowledgeCard>) -> Self {
        Self { cards, threshold: 2.0 }
    }
    pub fn with_threshold(mut self, t: f32) -> Self {
        self.threshold = t;
        self
    }

    pub fn select(&self, prompt: &str, token_budget: usize) -> Vec<&KnowledgeCard> {
        let prompt_lower = prompt.to_lowercase();
        let mut scored: Vec<(f32, &KnowledgeCard)> = self
            .cards
            .iter()
            .map(|c| {
                let mut score = 0.0f32;
                for k in &c.keywords {
                    let kl = k.to_lowercase();
                    if kl.contains(' ') {
                        if prompt_lower.contains(&kl) {
                            score += 2.0;
                        }
                    } else if prompt_lower.split(|ch: char| !ch.is_alphanumeric()).any(|w| w == kl) {
                        score += 1.0;
                    }
                }
                (score, c)
            })
            .filter(|(s, _)| *s >= self.threshold)
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut used = 0;
        let mut out = Vec::new();
        for (_, c) in scored {
            if used + c.token_cost > token_budget && !out.is_empty() {
                break;
            }
            out.push(c);
            used += c.token_cost;
            if used >= token_budget {
                break;
            }
        }
        out
    }
}

pub fn render_knowledge_block(cards: &[&KnowledgeCard]) -> String {
    if cards.is_empty() {
        return String::new();
    }
    let mut s = String::from("## Reference\n");
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

    fn card(topic: &str, kws: &[&str], cost: usize) -> KnowledgeCard {
        KnowledgeCard {
            topic: topic.into(),
            keywords: kws.iter().map(|s| s.to_string()).collect(),
            token_cost: cost,
            body: format!("### {topic}\nbody"),
            source_path: None,
        }
    }

    #[test]
    fn threshold_2_requires_multi_word_or_two_singles() {
        let inj = KnowledgeInjector::new(vec![
            card("A", &["binary", "search"], 100), // two singles = 2.0
            card("B", &["unrelated"], 100),
        ]);
        let sel = inj.select("we need a binary search algorithm", 200);
        assert_eq!(sel.len(), 1);
        assert_eq!(sel[0].topic, "A");
    }

    #[test]
    fn multi_word_phrase_matches_two_points() {
        let inj = KnowledgeInjector::new(vec![card("A", &["binary search"], 100)]);
        let sel = inj.select("how does binary search work", 200);
        assert_eq!(sel.len(), 1);
    }

    #[test]
    fn frontmatter_parses() {
        let md = "---\ntopic: Binary Search\nkeywords: [binary, search]\ntoken_cost: 90\n---\n### Body\nstuff\n";
        let c = parse_knowledge_markdown(md, None).unwrap();
        assert_eq!(c.topic, "Binary Search");
        assert_eq!(c.token_cost, 90);
    }

    #[test]
    fn empty_render_when_no_selection() {
        assert_eq!(render_knowledge_block(&[]), "");
    }
}
