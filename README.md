# PDF → PaddleOCR pipeline (test project)

Petit projet modulaire pour tester la pipeline OCR décrite par l'utilisateur :

- rendering PDF → images (PyMuPDF)
- preprocessing (OpenCV)
- layout detection (PP-Structure)
- OCR (PaddleOCR)
- simple reading-order + JSON output

Install requirements (prefer a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Usage:

```bash
python run_test.py path/to/input.pdf --out out.json
```

Notes:

- This project expects `paddleocr` installed and usable on your platform. For GPU usage, configure Paddle appropriately.
- The layout and OCR wrappers are thin; extend `pdf_parser/layout.py` and `pdf_parser/ocr_engine.py` to tune model parameters.
