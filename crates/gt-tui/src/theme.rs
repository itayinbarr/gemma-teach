use ratatui::style::Color;

pub const ACCENT: Color = Color::Rgb(255, 184, 108); // warm amber
pub const SUCCESS: Color = Color::Rgb(80, 250, 123);
pub const ERROR: Color = Color::Rgb(255, 85, 85);
pub const MUTED: Color = Color::Rgb(120, 120, 130);
pub const STREAM: Color = Color::Rgb(220, 220, 230);

pub const SPINNER_FRAMES: &[&str] = &[
    "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
];
