import tempfile
import time
from pathlib import Path

import cv2
import fitz
import numpy as np
import streamlit as st
from paddleocr import PPStructureV3
from PIL import Image


def uploaded_to_pages(uploaded_file, dpi: int, max_pages: int) -> list[np.ndarray]:
    data = uploaded_file.read()
    pages: list[np.ndarray] = []

    if uploaded_file.type == 'application/pdf' or uploaded_file.name.lower().endswith('.pdf'):
        doc = fitz.open(stream=data, filetype='pdf')
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(dpi=dpi)
            arr = np.frombuffer(pix.samples, dtype=np.uint8)
            arr = arr.reshape((pix.height, pix.width, pix.n))
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            pages.append(arr)
    else:
        from io import BytesIO
        image = Image.open(BytesIO(data)).convert('RGB')
        pages.append(np.array(image))

    return pages


def preprocess_page(rgb_img: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    denoise = cv2.bilateralFilter(gray, 9, 75, 75)
    binary = cv2.threshold(denoise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return {
        'gray': gray,
        'denoise': denoise,
        'binary': binary,
    }


def save_image(path: Path, img: np.ndarray) -> None:
    if img.ndim == 2:
        cv2.imwrite(str(path), img)
    else:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)


@st.cache_resource
def load_pipeline() -> PPStructureV3:
    return PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )


def read_text_files(folder: Path, suffix: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for file in sorted(folder.glob(f'**/*{suffix}')):
        try:
            text = file.read_text(encoding='utf-8', errors='replace')
            items.append((str(file.name), text))
        except Exception as exc:
            items.append((str(file.name), f'Could not read file: {exc}'))
    return items


def image_outputs(folder: Path) -> list[Path]:
    exts = {'.png', '.jpg', '.jpeg'}
    return [p for p in sorted(folder.glob('**/*')) if p.suffix.lower() in exts]


def run_ppstructure(pipeline: PPStructureV3, input_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = list(pipeline.predict(input=str(input_path)))

    for res in results:
        if hasattr(res, 'save_to_json'):
            res.save_to_json(save_path=str(output_dir))
        if hasattr(res, 'save_to_markdown'):
            res.save_to_markdown(save_path=str(output_dir))

    preview = []
    for idx, res in enumerate(results):
        preview.append({
            'index': idx,
            'type': type(res).__name__,
            'repr': str(res)[:5000],
        })

    return {
        'count': len(results),
        'preview': preview,
        'json_files': read_text_files(output_dir, '.json'),
        'md_files': read_text_files(output_dir, '.md'),
        'images': image_outputs(output_dir),
    }


st.set_page_config(page_title='PDF/OCR Pipeline Debug', layout='wide')
st.title('Simple PDF -> OCR Pipeline Visualizer')
st.caption('Upload a PDF or image and inspect each processing step.')

with st.sidebar:
    st.header('Settings')
    dpi = st.slider('PDF render DPI', min_value=120, max_value=400, value=220, step=20)
    max_pages = st.slider('Max pages', min_value=1, max_value=20, value=3, step=1)
    run_on_preprocessed = st.checkbox('Run PPStructureV3 on binary image', value=False)

uploaded = st.file_uploader('Input file', type=['pdf', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'])
run_btn = st.button('Run pipeline', type='primary', disabled=uploaded is None)

if run_btn and uploaded is not None:
    base_tmp = Path(tempfile.mkdtemp(prefix='ppstruct_debug_'))

    t0 = time.perf_counter()
    pages = uploaded_to_pages(uploaded, dpi=dpi, max_pages=max_pages)
    t1 = time.perf_counter()

    st.success(f'Loaded {len(pages)} page(s) in {t1 - t0:.2f}s')

    pipeline = load_pipeline()

    for i, page_rgb in enumerate(pages, start=1):
        with st.expander(f'Page {i}', expanded=(i == 1)):
            pre = preprocess_page(page_rgb)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown('**1) Original**')
                st.image(page_rgb, use_container_width=True)
            with c2:
                st.markdown('**2) Gray**')
                st.image(pre['gray'], clamp=True, use_container_width=True)
            with c3:
                st.markdown('**3) Denoised**')
                st.image(pre['denoise'], clamp=True, use_container_width=True)
            with c4:
                st.markdown('**4) Binary (Otsu)**')
                st.image(pre['binary'], clamp=True, use_container_width=True)

            page_dir = base_tmp / f'page_{i:02d}'
            page_dir.mkdir(parents=True, exist_ok=True)

            source_img = pre['binary'] if run_on_preprocessed else page_rgb
            if source_img.ndim == 2:
                source_for_model = cv2.cvtColor(source_img, cv2.COLOR_GRAY2RGB)
            else:
                source_for_model = source_img

            input_path = page_dir / 'input_for_model.png'
            save_image(input_path, source_for_model)

            t_start_model = time.perf_counter()
            model_out = run_ppstructure(pipeline, input_path=input_path, output_dir=page_dir / 'outputs')
            t_end_model = time.perf_counter()

            st.markdown(f'**5) Model execution:** {t_end_model - t_start_model:.2f}s')
            st.write(f"Result objects: {model_out['count']}")

            st.markdown('**6) Result object preview**')
            st.json(model_out['preview'])

            if model_out['images']:
                st.markdown('**7) Saved result images**')
                img_cols = st.columns(min(3, len(model_out['images'])))
                for idx, p in enumerate(model_out['images'][:6]):
                    with img_cols[idx % len(img_cols)]:
                        st.image(str(p), caption=p.name, use_container_width=True)

            if model_out['json_files']:
                st.markdown('**8) JSON outputs**')
                for name, text in model_out['json_files'][:3]:
                    st.code(text, language='json')
                    st.caption(name)

            if model_out['md_files']:
                st.markdown('**9) Markdown outputs**')
                for name, text in model_out['md_files'][:3]:
                    st.code(text, language='markdown')
                    st.caption(name)

            st.caption(f'Temporary folder: {page_dir}')

    st.info(f'All temporary outputs stored under: {base_tmp}')
