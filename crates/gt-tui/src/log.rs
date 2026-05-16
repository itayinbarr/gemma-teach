//! Redirect the process's stderr fd to a log file for the lifetime of the TUI.
//!
//! Ratatui's alternate screen does not isolate file descriptors — both stdout
//! and stderr write to the same TTY. Anything that writes to stderr while the
//! TUI is active corrupts the rendered layout (the rescaling escape hatch
//! works only because ratatui clears the screen on `Event::Resize`).
//!
//! The offending writers are not in our Rust code — they are the C library
//! `llama.cpp` (loader / Metal / context messages) and our own
//! `tracing_subscriber` which defaults to stderr at warn level. Routing
//! `tracing` separately catches the latter; the former is reachable only by
//! redirecting the *fd*. We do both: `send_logs_to_tracing` upstream, and
//! this guard below as belt-and-suspenders for any C-side writer that bypasses
//! the log callback (ggml-metal has historically used direct `fprintf`).

use std::fs::OpenOptions;
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::Path;

/// Holds the original stderr fd; restores it on drop.
pub struct StderrRedirect {
    saved: RawFd,
}

impl StderrRedirect {
    /// Open `path` (append + create) and `dup2` it onto fd 2 (stderr).
    pub fn to_file(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)?;

        // SAFETY: fd 2 is the process's stderr; `dup`/`dup2` are async-signal
        // safe and have no rust-side aliasing concerns.
        let saved = unsafe { libc::dup(libc::STDERR_FILENO) };
        if saved < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let target = file.as_raw_fd();
        let rc = unsafe { libc::dup2(target, libc::STDERR_FILENO) };
        if rc < 0 {
            let err = std::io::Error::last_os_error();
            unsafe { libc::close(saved) };
            return Err(err);
        }
        // The file object can be dropped now — fd 2 holds its own reference
        // via dup2.
        drop(file);
        Ok(Self { saved })
    }
}

impl Drop for StderrRedirect {
    fn drop(&mut self) {
        unsafe {
            libc::dup2(self.saved, libc::STDERR_FILENO);
            libc::close(self.saved);
        }
    }
}
