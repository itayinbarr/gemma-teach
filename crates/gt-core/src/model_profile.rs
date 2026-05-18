use serde::{Deserialize, Serialize};

/// Per-model tuning knobs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub name: String,
    pub context_window: usize,
    pub max_tokens: usize,
    pub thinking_budget: usize,
    pub skill_token_budget: usize,
    pub knowledge_token_budget: usize,
    pub turn_cap: u32,
    pub temperature: f32,
    pub max_consecutive_corrections: u32,
}

impl ModelProfile {
    /// Defaults tuned for Gemma 4 E2B (small, ~2 B effective parameters).
    pub fn gemma_4_e2b() -> Self {
        Self {
            name: "gemma-4-E2B-it-Q4_K_M".into(),
            context_window: 8192,
            max_tokens: 1024,
            thinking_budget: 1024,
            skill_token_budget: 200,
            knowledge_token_budget: 150,
            turn_cap: 15,
            temperature: 0.4,
            max_consecutive_corrections: 2,
        }
    }

    /// Permissive profile for mocked tests where we don't care about caps.
    pub fn test_default() -> Self {
        Self {
            name: "mock".into(),
            context_window: 32_768,
            max_tokens: 4096,
            thinking_budget: 4096,
            skill_token_budget: 500,
            knowledge_token_budget: 500,
            turn_cap: 50,
            temperature: 0.0,
            max_consecutive_corrections: 2,
        }
    }
}

impl Default for ModelProfile {
    fn default() -> Self {
        Self::gemma_4_e2b()
    }
}
