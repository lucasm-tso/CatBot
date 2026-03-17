# PDF to Ollama Markdown

Simple script to:
1. read a PDF,
2. run OCR with PaddleOCR (orientation enabled) or with a Qwen/Ollama vision model,
3. send extracted page text to an Ollama model (`qwen3.5:35b` by default),
4. write the model result to a `.md` file.

## Install

```bash
cd /home/tso-ia/Documents/CatBot/pdf_to_ollama_md
pip install -r requirements.txt
```

## Basic usage

```bash
python run_pdf_to_md.py /path/to/file.pdf \
  --ollama-url http://localhost:7869 \
  --model qwen3.5:35b
```

This writes `./<pdf_name>_ollama.md` by default.

## With custom prompt and output

```bash
python run_pdf_to_md.py /path/to/file.pdf \
  --ollama-url http://localhost:7869 \
  --model qwen3.5:35b \
  --prompt "Extract key points and list action items." \
  --output /path/to/result.md
```

## Process selected pages only

```bash
python run_pdf_to_md.py /path/to/file.pdf \
  --ollama-url http://localhost:7869 \
  --pages 1,3-5
```

## OCR with Qwen model (no PaddleOCR)

Use a vision-capable Ollama model for OCR:

```bash
python run_pdf_to_md.py /path/to/file.pdf \
  --ollama-url http://localhost:7869 \
  --ocr-engine qwen \
  --model qwen2.5vl:7b
```

Use one model for OCR and another for final reasoning:

```bash
python run_pdf_to_md.py /path/to/file.pdf \
  --ollama-url http://localhost:7869 \
  --ocr-engine qwen \
  --ocr-model qwen2.5vl:7b \
  --model qwen3.5:35b
```

## Stream generation (including thinking when available)

```bash
python run_pdf_to_md.py /path/to/file.pdf \
  --ollama-url http://localhost:7869 \
  --ocr-engine qwen \
  --ocr-model qwen2.5vl:7b \
  --model qwen3.5:35b \
  --stream \
  --show-thinking
```

Streaming output is printed live to `stderr` while the final markdown file is still written at the end.

## Debug metrics (timings + tool usage)

Add `--debug-metrics` to print detailed metrics to `stderr`, and also include metrics in the final JSON summary:

```bash
python run_pdf_to_md.py /path/to/file.pdf \
  --ollama-url http://localhost:7869 \
  --ocr-engine qwen \
  --guided-zoom \
  --stream --show-thinking \
  --debug-metrics
```

Metrics include totals and per-part timings/counters (examples):

- `total_runtime_seconds`
- `page_total_seconds`, `qwen_coarse_page_seconds`
- `zoom_proposal_calls`, `zoom_regions_proposed`, `zoom_regions_used`
- `zoom_crop_ocr_calls`, `zoom_crop_ocr_seconds`
- `qwen_ocr_calls`, `qwen_ocr_seconds`, `qwen_ocr_timeouts`
- `final_reasoning_calls`, `final_reasoning_seconds`

## Model-guided zoom OCR (bounding boxes)

Enable a second OCR pass where the model proposes uncertain regions, then the script renders high-DPI crops for those boxes and refines the page text:

```bash
python run_pdf_to_md.py /path/to/file.pdf \
  --ollama-url http://localhost:7869 \
  --ocr-engine qwen \
  --ocr-model qwen2.5vl:7b \
  --guided-zoom \
  --zoom-crop-dpi 450 \
  --max-zoom-requests-per-page 3
```

Useful tuning flags:

- `--zoom-crop-dpi`: crop render quality (higher = better detail, slower)
- `--max-zoom-requests-per-page`: cap model-requested crops per page
- `--zoom-min-box-size`: reject tiny boxes
- `--zoom-max-total-area`: cap total zoomed area per page (normalized 0..1)

## Prompt from file

```bash
python run_pdf_to_md.py /path/to/file.pdf \
  --ollama-url http://localhost:7869 \
  --prompt-file /path/to/prompt.txt
```

## Notes

- Your docker compose maps Ollama to `7869`, so `--ollama-url http://localhost:7869` is likely needed.
- If OCR misses text, try higher `--dpi` (for example `--dpi 300`).
- PaddleOCR language defaults to `en`; change with `--lang fr` or others.
- `--ocr-engine qwen` requires a vision model in Ollama (for example `qwen2.5vl:*`).
- Some models do not expose separate thinking tokens; `--show-thinking` only prints them when provided by Ollama events.
