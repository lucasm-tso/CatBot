# Test Web App

Simple Streamlit UI to visualize a document parsing pipeline step by step.

## Run

```bash
pip install -r requirements.txt
streamlit run test_webapp/app.py
```

## What You See

1. Original page/image
2. Grayscale
3. Denoised image
4. Binary image (Otsu)
5. `PPStructureV3` execution
6. Raw result object preview
7. Saved output images (if generated)
8. Saved JSON outputs
9. Saved Markdown outputs

Use sidebar settings to control PDF DPI, max pages, and whether to run the model on binary images.
