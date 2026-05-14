//! TUI renderer + event loop.

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use gt_core::session_event::StepState;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;
use std::time::{Duration, Instant};

use crate::app::{App, AppMode, FormField, StudentAddForm, StudentEditForm};
use crate::slash::{parse, Slash};
use crate::theme;

pub async fn run(mut app: App) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut term = Terminal::new(backend)?;

    let tick = Duration::from_millis(33);
    let mut last_tick = Instant::now();
    let result: Result<()> = loop {
        // Drain async events into app state.
        app.drain_events();
        app.tick_count = app.tick_count.wrapping_add(1);

        // Draw.
        term.draw(|f| draw(f, &app))?;

        // Poll input.
        let timeout = tick
            .checked_sub(last_tick.elapsed())
            .unwrap_or(Duration::from_millis(0));
        if event::poll(timeout)? {
            match event::read()? {
                Event::Key(k) => handle_key(&mut app, k),
                Event::Resize(_, _) => {}
                _ => {}
            }
        }
        if last_tick.elapsed() >= tick {
            last_tick = Instant::now();
        }
        if app.should_quit {
            break Ok(());
        }
    };

    disable_raw_mode().ok();
    term.backend_mut().execute(LeaveAlternateScreen).ok();
    term.show_cursor().ok();
    result
}

fn handle_key(app: &mut App, k: KeyEvent) {
    // global shortcuts
    if k.modifiers.contains(KeyModifiers::CONTROL) && matches!(k.code, KeyCode::Char('c')) {
        app.should_quit = true;
        return;
    }

    if matches!(app.mode, AppMode::Help) && matches!(k.code, KeyCode::Esc) {
        app.mode = AppMode::Idle;
        return;
    }
    match &mut app.mode {
        AppMode::Idle | AppMode::FlowActive | AppMode::Help => match k.code {
            KeyCode::Char(c) => {
                app.input.push(c);
            }
            KeyCode::Backspace => {
                app.input.pop();
            }
            KeyCode::Tab => {
                if !app.steps.is_empty() {
                    app.selected_step = (app.selected_step + 1) % app.steps.len();
                }
            }
            KeyCode::BackTab => {
                if !app.steps.is_empty() {
                    if app.selected_step == 0 {
                        app.selected_step = app.steps.len() - 1;
                    } else {
                        app.selected_step -= 1;
                    }
                }
            }
            KeyCode::Esc => {
                app.input.clear();
            }
            KeyCode::Enter => {
                let line = std::mem::take(&mut app.input);
                submit_slash(app, &line);
            }
            _ => {}
        },
        AppMode::StudentEditModal(form) => match k.code {
            KeyCode::Esc => app.mode = AppMode::Idle,
            KeyCode::Enter if k.modifiers.contains(KeyModifiers::CONTROL) => {
                let name = form.name.clone();
                let notes = form.notes.trim().to_string();
                if notes.is_empty() {
                    app.log("Edit notes are empty — type Esc to cancel.");
                    return;
                }
                app.start_student_edit(name, notes);
            }
            KeyCode::Backspace => {
                form.notes.pop();
            }
            KeyCode::Enter => form.notes.push('\n'),
            KeyCode::Char(c) => form.notes.push(c),
            _ => {}
        },
        AppMode::StudentAddModal(form) => match k.code {
            KeyCode::Esc => {
                app.mode = AppMode::Idle;
            }
            KeyCode::Tab => {
                form.focus = match form.focus {
                    FormField::Name => FormField::Description,
                    FormField::Description => FormField::Name,
                };
            }
            KeyCode::Enter if k.modifiers.contains(KeyModifiers::CONTROL) => {
                let name = form.name.trim().to_string();
                let description = form.description.trim().to_string();
                if name.is_empty() || description.is_empty() {
                    app.log("Both Name and Description are required.");
                    return;
                }
                app.start_student_add(name, description);
            }
            KeyCode::Enter => match form.focus {
                FormField::Description => form.description.push('\n'),
                FormField::Name => {
                    form.focus = FormField::Description;
                }
            },
            KeyCode::Backspace => match form.focus {
                FormField::Name => {
                    form.name.pop();
                }
                FormField::Description => {
                    form.description.pop();
                }
            },
            KeyCode::Char(c) => match form.focus {
                FormField::Name => form.name.push(c),
                FormField::Description => form.description.push(c),
            },
            _ => {}
        },
    }
}

fn submit_slash(app: &mut App, line: &str) {
    if line.trim().is_empty() {
        return;
    }
    match parse(line) {
        Some(Slash::StudentAdd) => {
            app.mode = AppMode::StudentAddModal(StudentAddForm::default());
        }
        Some(Slash::ClassPlan { pdf }) => {
            if !pdf.exists() {
                app.log(format!("/class-plan: PDF not found at {}", pdf.display()));
            } else {
                app.start_class_plan(pdf);
            }
        }
        Some(Slash::StudentEdit { name }) => {
            // Verify the student exists.
            let dir = app.root.join("students").join(crate::app::slug_or_self(&name));
            if !dir.exists() {
                app.log(format!(
                    "/student-edit: no student '{name}' under {}",
                    app.root.join("students").display()
                ));
            } else {
                app.mode = AppMode::StudentEditModal(StudentEditForm {
                    name: name.clone(),
                    notes: String::new(),
                });
            }
        }
        Some(Slash::Help) => {
            app.mode = AppMode::Help;
        }
        Some(Slash::Quit) => {
            app.should_quit = true;
        }
        Some(Slash::Unknown(s)) => {
            app.log(s);
        }
        None => {
            app.log(format!("not a command: '{}'", line.trim()));
        }
    }
}

fn draw(f: &mut ratatui::Frame, app: &App) {
    let area = f.area();
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // header
            Constraint::Min(5),    // body
            Constraint::Length(3), // input
        ])
        .split(area);

    draw_header(f, outer[0], app);
    draw_body(f, outer[1], app);
    draw_input(f, outer[2], app);

    if let AppMode::StudentAddModal(form) = &app.mode {
        draw_modal(f, area, form);
    }
    if let AppMode::StudentEditModal(form) = &app.mode {
        draw_edit_modal(f, area, form);
    }
    if matches!(app.mode, AppMode::Help) {
        draw_help_modal(f, area);
    }
}

fn draw_edit_modal(f: &mut ratatui::Frame, area: Rect, form: &StudentEditForm) {
    let centered = centered_rect(75, 70, area);
    f.render_widget(ratatui::widgets::Clear, centered);
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(2),
        ])
        .split(centered);

    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            "/student-edit ",
            Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::raw(form.name.clone()),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::MUTED)),
    );
    f.render_widget(title, layout[0]);

    let body = Paragraph::new(form.notes.as_str())
        .block(
            Block::default()
                .title(" Edit notes (free text) ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::ACCENT)),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(body, layout[1]);

    let hint = Paragraph::new(Line::from(vec![
        Span::styled(
            "Ctrl-Enter",
            Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" submit    "),
        Span::styled("Esc", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
        Span::raw(" cancel"),
    ]));
    f.render_widget(hint, layout[2]);
}

fn draw_header(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let flow = app
        .flow_name
        .as_deref()
        .unwrap_or("(no flow running)");
    let elapsed = app
        .flow_started_at
        .map(|t| t.elapsed().as_secs())
        .unwrap_or(0);
    let backend_label = if app.backend.is_loaded() { "ready" } else { "lazy-load" };
    let title = Line::from(vec![
        Span::styled(
            "Gemma Teach",
            Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::raw("   "),
        Span::styled(flow, Style::default().fg(theme::STREAM)),
        Span::raw(format!("   {}:{:02}   ", elapsed / 60, elapsed % 60)),
        Span::styled(format!("model: {}", backend_label), Style::default().fg(theme::MUTED)),
    ]);
    let last_msg = app
        .messages
        .back()
        .map(|s| Line::from(s.as_str()))
        .unwrap_or_default();
    let p = Paragraph::new(vec![title, last_msg]);
    f.render_widget(p, area);
}

fn draw_body(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
        .split(area);
    draw_tasks(f, cols[0], app);
    draw_detail(f, cols[1], app);
}

fn draw_tasks(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let mut lines: Vec<Line> = Vec::new();
    for (i, step) in app.steps.iter().enumerate() {
        let (glyph, color) = match step.state {
            StepState::Queued => ("◌", theme::MUTED),
            StepState::Running => (theme::SPINNER_FRAMES
                [(app.tick_count as usize) % theme::SPINNER_FRAMES.len()], theme::ACCENT),
            StepState::Streaming => (theme::SPINNER_FRAMES
                [(app.tick_count as usize) % theme::SPINNER_FRAMES.len()], theme::ACCENT),
            StepState::Done => ("✔", theme::SUCCESS),
            StepState::Failed => ("✖", theme::ERROR),
        };
        let mut spans = vec![
            Span::styled(format!(" {} ", glyph), Style::default().fg(color)),
            Span::styled(
                step.name.clone(),
                if i == app.selected_step {
                    Style::default()
                        .fg(theme::ACCENT)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme::STREAM)
                },
            ),
            Span::raw("  "),
            Span::styled(format!("[{}]", step.kind), Style::default().fg(theme::MUTED)),
        ];
        if let Some(t) = &step.last_tool {
            spans.push(Span::raw("  "));
            spans.push(Span::styled(format!("tool: {}", t), Style::default().fg(theme::MUTED)));
        }
        lines.push(Line::from(spans));
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no flow running — type a / command below",
            Style::default().fg(theme::MUTED),
        )));
    }
    let block = Block::default()
        .title(" Tasks ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::MUTED));
    f.render_widget(Paragraph::new(lines).block(block).wrap(Wrap { trim: false }), area);
}

fn draw_detail(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(" Detail ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::MUTED));
    let mut lines: Vec<Line> = Vec::new();
    if let Some(step) = app.current_step() {
        lines.push(Line::from(vec![
            Span::styled("step: ", Style::default().fg(theme::MUTED)),
            Span::styled(
                step.name.clone(),
                Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                format!("({})", step_state_label(step.state)),
                Style::default().fg(theme::MUTED),
            ),
        ]));
        if !step.artifacts.is_empty() {
            lines.push(Line::from(Span::styled(
                "artifacts:",
                Style::default().fg(theme::MUTED),
            )));
            for a in &step.artifacts {
                lines.push(Line::from(format!("  {}", a)));
            }
        }
        if !step.streaming_text.is_empty() {
            lines.push(Line::from(""));
            for line in step.streaming_text.lines().rev().take(40).collect::<Vec<_>>().iter().rev() {
                lines.push(Line::from(Span::styled(
                    (*line).to_string(),
                    Style::default().fg(theme::STREAM),
                )));
            }
        }
    } else {
        for m in app.messages.iter().rev().take(20).collect::<Vec<_>>().iter().rev() {
            lines.push(Line::from(Span::styled(
                m.as_str().to_string(),
                Style::default().fg(theme::STREAM),
            )));
        }
    }
    f.render_widget(Paragraph::new(lines).block(block).wrap(Wrap { trim: false }), area);
}

fn step_state_label(s: StepState) -> &'static str {
    match s {
        StepState::Queued => "queued",
        StepState::Running => "running",
        StepState::Streaming => "streaming",
        StepState::Done => "done",
        StepState::Failed => "failed",
    }
}

fn draw_input(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::ACCENT));
    let line = Line::from(vec![
        Span::styled("» ", Style::default().fg(theme::ACCENT)),
        Span::raw(&app.input),
        Span::styled("▎", Style::default().fg(theme::ACCENT)),
    ]);
    f.render_widget(Paragraph::new(line).block(block), area);
}

fn draw_modal(f: &mut ratatui::Frame, area: Rect, form: &StudentAddForm) {
    let centered = centered_rect(75, 80, area);
    f.render_widget(ratatui::widgets::Clear, centered);
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // name field
            Constraint::Min(5),    // description
            Constraint::Length(2), // hint
        ])
        .split(centered);

    let name_focused = matches!(form.focus, FormField::Name);
    let desc_focused = matches!(form.focus, FormField::Description);

    let name_block = Block::default()
        .title(" /student-add — Name ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if name_focused {
            theme::ACCENT
        } else {
            theme::MUTED
        }));
    let name_p = Paragraph::new(form.name.as_str()).block(name_block);
    f.render_widget(name_p, layout[0]);

    let desc_block = Block::default()
        .title(" Description (free-text — interests, hobbies, favorite media) ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if desc_focused {
            theme::ACCENT
        } else {
            theme::MUTED
        }));
    let desc_p = Paragraph::new(form.description.as_str())
        .block(desc_block)
        .wrap(Wrap { trim: false });
    f.render_widget(desc_p, layout[1]);

    let hint = Paragraph::new(Line::from(vec![
        Span::styled("Tab", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
        Span::raw(" switch field    "),
        Span::styled("Ctrl-Enter", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
        Span::raw(" submit    "),
        Span::styled("Esc", Style::default().fg(theme::ACCENT).add_modifier(Modifier::BOLD)),
        Span::raw(" cancel"),
    ]));
    f.render_widget(hint, layout[2]);
}

fn draw_help_modal(f: &mut ratatui::Frame, area: Rect) {
    let centered = centered_rect(60, 60, area);
    f.render_widget(ratatui::widgets::Clear, centered);
    let block = Block::default()
        .title(" Help ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::ACCENT));
    let body = vec![
        Line::from("Slash commands:"),
        Line::from("  /student-add                   add a student"),
        Line::from("  /class-plan <pdf>              build a lesson from a PDF"),
        Line::from("  /student-edit <name>           update a student's profile"),
        Line::from("  /help                          show this screen"),
        Line::from("  /quit                          exit"),
        Line::from(""),
        Line::from("Keys:"),
        Line::from("  Tab / Shift-Tab                cycle the focused task"),
        Line::from("  Ctrl-C                         quit immediately"),
        Line::from(""),
        Line::from("Press Esc to dismiss this screen, then any key to continue."),
    ];
    f.render_widget(Paragraph::new(body).block(block).wrap(Wrap { trim: false }), centered);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
