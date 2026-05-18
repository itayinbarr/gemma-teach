use anyhow::Result;
use clap::Parser;

mod app;
mod backend_select;
mod log;
mod slash;
mod theme;
mod ui;

#[derive(Parser, Debug)]
#[command(name = "gemma-teach", about = "Claude Code–style harness for teachers")]
struct Args {
    /// Path to the GGUF model. Overrides the default cache location.
    #[arg(long, env = "GEMMA_TEACH_MODEL")]
    model: Option<std::path::PathBuf>,

    /// Root directory for the class notebook.
    #[arg(long, env = "GEMMA_TEACH_ROOT")]
    root: Option<std::path::PathBuf>,

    /// Backend to use. "llama" (default) loads Gemma 4 E2B locally; "mock" uses a
    /// scripted backend for development; "echo" is a chatty no-op.
    #[arg(long, env = "GEMMA_TEACH_BACKEND", default_value = "llama")]
    backend: String,

    /// Download the model and exit. First-launch convenience.
    #[arg(long)]
    download_only: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("GEMMA_TEACH_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let root = args
        .root
        .clone()
        .or_else(|| dirs::home_dir().map(|h| h.join("GemmaTeach")))
        .unwrap_or_else(|| std::path::PathBuf::from("./GemmaTeach"));
    tokio::fs::create_dir_all(&root).await.ok();

    if args.download_only {
        return backend_select::download_default_model().await;
    }

    let log_path = log_file_path();
    if let Some(parent) = log_path.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }
    eprintln!("Logs: {}", log_path.display());

    let backend = backend_select::build(&args.backend, args.model.clone()).await?;
    let app = app::App::new(root, backend);
    ui::run(app, log_path).await
}

fn log_file_path() -> std::path::PathBuf {
    let base = dirs::home_dir()
        .map(|h| h.join(".gemma-teach").join("logs"))
        .unwrap_or_else(|| std::path::PathBuf::from(".gemma-teach/logs"));
    let day = chrono::Local::now().format("%Y-%m-%d").to_string();
    base.join(format!("gemma-teach-{day}.log"))
}
