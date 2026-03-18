# DataSet Adapter - Batch PDF to Markdown

## Overview
This app recursively scans a folder for PDF files, converts each PDF to markdown using OCR + Ollama, and writes:
- `<pdf_stem>_ollama.md` beside each source PDF
- `<pdf_stem>_summary.md` beside each source PDF
- `_batch_conversion_summary_<YYYYMMDD_HHMMSS>.md` in the scanned root folder

The implementation is standalone and self-contained inside `DataSet_adapter/src`.
It uses structured internal modules for OCR, Ollama calls, markdown generation, batching, and reporting.

## Setup
```bash
cd /home/tso-ia/Documents/CatBot/DataSet_adapter
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Default runtime profile
- Ollama URL: `http://localhost:7869`
- Model: `qwen3.5:27b`
- OCR engine: `qwen`
- Stream: enabled
- Show thinking: enabled
- Guided zoom: enabled

## Notes
- Batch mode is sequential by design.
- Failures do not stop the batch; they are recorded in summaries.
- Large PDFs can take a long time, especially with guided zoom and large models.
