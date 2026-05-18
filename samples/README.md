# samples

Reusable inputs for testing Gemma Teach end-to-end against real Gemma 4.

## chapters/

Plain-text "OCR'd chapter" inputs for `/class-plan`. The trace recorder treats
these as if they came out of Tesseract.

- `photosynthesis.txt` — middle-school biology chapter; pairs well with a class
  that has students interested in art (Studio Ghibli, drawing) and building
  (Minecraft, LEGO) so the tailoring step has meaningfully different framings
  to produce.

## Using a sample with `record_trace`

```sh
cargo run -p gt-flows --example record_trace --features smoke --release -- \
    class-plan --source-txt samples/chapters/photosynthesis.txt \
    --notebook ~/GemmaTeach --out traces/class-plan.jsonl
```
