default:
    @just --list

run:
    cargo run -p gt-tui

dev:
    RUST_LOG=info,gt_core=debug,gt_flows=debug,gt_tui=debug cargo run -p gt-tui

download-model:
    cargo run -p gt-tui -- --download-only

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

lint:
    cargo clippy --workspace --all-targets -- -D warnings

test:
    cargo test --workspace

test-smoke:
    GEMMA_TEACH_SMOKE=1 cargo test --workspace --test smoke_real_model -- --nocapture

deps-check:
    @which tesseract >/dev/null && echo "tesseract: ok" || echo "tesseract: MISSING (brew install tesseract)"
    @which pdftoppm >/dev/null && echo "pdftoppm:  ok" || echo "pdftoppm:  MISSING (brew install poppler)"
    @which typst >/dev/null && echo "typst:     ok" || echo "typst:     MISSING (brew install typst)"

check:
    cargo check --workspace --all-targets
