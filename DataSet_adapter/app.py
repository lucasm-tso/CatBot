#!/usr/bin/env python3
"""Streamlit app for recursive PDF to Markdown conversion."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from src.batch_service import run_batch_conversion
from src.config import AppConfig
from src.discovery import discover_pdfs

if TYPE_CHECKING:
    pass


class _StderrCapture:
    """Redirige sys.stderr vers un placeholder Streamlit pour afficher le thinking en direct."""

    _UPDATE_EVERY = 150  # nombre de caractères entre chaque rafraîchissement UI

    def __init__(self, placeholder: st.delta_generator.DeltaGenerator) -> None:
        self._placeholder = placeholder
        self._chunks: list[str] = []
        self._since_update = 0

    def write(self, text: str) -> int:
        if text:
            self._chunks.append(text)
            self._since_update += len(text)
            if self._since_update >= self._UPDATE_EVERY:
                self._flush_ui()
        return len(text)

    def flush(self) -> None:
        self._flush_ui()

    def _flush_ui(self) -> None:
        self._since_update = 0
        combined = "".join(self._chunks)
        # Afficher les 4000 derniers caractères pour éviter un texte trop long
        visible = combined[-4000:] if len(combined) > 4000 else combined
        self._placeholder.code(visible, language="text")

    def getvalue(self) -> str:
        return "".join(self._chunks)


def _generated_paths_for_pdf(pdf_path: Path) -> tuple[Path, Path]:
    return (
        pdf_path.with_name(f"{pdf_path.stem}_ollama.md"),
        pdf_path.with_name(f"{pdf_path.stem}_summary.md"),
    )


def _has_generated_outputs(pdf_path: Path) -> bool:
    out_md, summary_md = _generated_paths_for_pdf(pdf_path)
    return out_md.exists() or summary_md.exists()


def _delete_generated_outputs(
    pdf_paths: list[Path],
    delete_ollama_md: bool,
    delete_summary_md: bool,
) -> tuple[int, list[str]]:
    deleted = 0
    errors: list[str] = []
    for pdf_path in pdf_paths:
        out_md, summary_md = _generated_paths_for_pdf(pdf_path)
        targets: list[Path] = []
        if delete_ollama_md:
            targets.append(out_md)
        if delete_summary_md:
            targets.append(summary_md)

        for target in targets:
            if not target.exists():
                continue
            try:
                target.unlink()
                deleted += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{target}: {exc}")
    return deleted, errors


st.set_page_config(page_title="Batch PDF to Markdown", layout="wide")
st.title("Batch PDF to Markdown Converter")
st.caption("Recursive conversion with OCR, Ollama, uncertainty summaries, and debug metrics.")

with st.sidebar:
    st.header("Runtime Parameters")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:7869")
    model = st.text_input("Model", value="qwen3.5:122b")
    ocr_engine = st.selectbox("OCR Engine", options=["qwen", "paddle"], index=0)
    stream = st.checkbox("Stream output", value=True)
    show_thinking = st.checkbox("Show thinking", value=True)
    guided_zoom = st.checkbox("Guided zoom", value=True)
    use_page_images_in_review = st.checkbox(
        "Use full page image in clean review",
        value=True,
        help="When enabled, the final page clean review sees the full page image in addition to OCR text.",
    )
    skip_generated = st.checkbox(
        "Skip already generated PDFs",
        value=True,
        help="Exclude PDFs that already have _ollama.md or _summary.md from the conversion queue.",
    )

    with st.expander("Advanced"):
        dpi = st.number_input("DPI", min_value=100, max_value=1200, value=600, step=50)
        review_batch_size = st.number_input(
            "Review batch size (pages)",
            min_value=1,
            max_value=8,
            value=1,
            step=1,
            help="Pages grouped per clean review call. Lower values are safer for very long documents.",
        )
        review_image_max_side = st.number_input(
            "Review image max side (px)",
            min_value=800,
            max_value=5000,
            value=2400,
            step=200,
            help="Image downscale limit sent to the review model. Higher improves details but costs more VRAM/time.",
        )
        connect_timeout = st.number_input("Connect timeout (s)", min_value=1, max_value=120, value=15)
        read_timeout = st.number_input("Read timeout (s)", min_value=30, max_value=7200, value=900)
        final_timeout = st.number_input("Final reasoning timeout (s)", min_value=30, max_value=1800, value=180)
        qwen_retries = st.number_input("Qwen retries", min_value=0, max_value=10, value=1)

root_input = st.text_input("Root folder to scan", value="/home/tso-ia/Documents/Data")
root_path = Path(root_input).expanduser()

scan_col, run_col, clean_col = st.columns(3)

if scan_col.button("Scan PDFs", width='stretch'):
    try:
        pdfs = discover_pdfs(root_path)
        st.session_state["scan_results"] = pdfs
    except Exception as exc:  # noqa: BLE001
        st.error(f"Scan failed: {exc}")

scan_results = st.session_state.get("scan_results", [])
if scan_results:
    already_generated = [p for p in scan_results if _has_generated_outputs(p)]
    pending_queue = [p for p in scan_results if p not in already_generated]

    st.success(f"Found {len(scan_results)} PDF files.")
    st.caption(
        f"Queue status: pending={len(pending_queue)} | already generated={len(already_generated)}"
    )
    st.dataframe({"pdf_path": [str(p) for p in scan_results]}, width='stretch')
else:
    already_generated = []
    pending_queue = []

with st.expander("Delete Generated Files"):
    if not scan_results:
        st.info("No scanned PDFs yet. Run Scan PDFs first.")
    elif not already_generated:
        st.info("No generated markdown files found for scanned PDFs.")
    else:
        _delete_mode = st.radio(
            "Delete file type",
            options=["Both (_ollama + _summary)", "Only _ollama.md", "Only _summary.md"],
            index=0,
            horizontal=True,
            key="delete_mode",
        )

        _generated_options = [str(p) for p in already_generated]
        _selected_for_delete = st.multiselect(
            "Choose PDFs to clean",
            options=_generated_options,
            default=_generated_options,
            key="delete_pdf_selection",
            help="Only generated markdown files next to these PDFs will be deleted.",
        )

        if clean_col.button("Delete Selected .md", width='stretch'):
            if not _selected_for_delete:
                st.warning("Select at least one PDF to delete generated files.")
            else:
                selected_paths = [Path(p) for p in _selected_for_delete]
                delete_ollama = _delete_mode != "Only _summary.md"
                delete_summary = _delete_mode != "Only _ollama.md"
                deleted_count, delete_errors = _delete_generated_outputs(
                    selected_paths,
                    delete_ollama_md=delete_ollama,
                    delete_summary_md=delete_summary,
                )
                if deleted_count:
                    st.success(f"Deleted {deleted_count} generated markdown files.")
                else:
                    st.info("No matching generated markdown files were found to delete.")
                if delete_errors:
                    st.error("\n".join(delete_errors[:10]))

if run_col.button("Run Conversion", width='stretch'):
    logs: list[str] = []
    progress_bar = st.progress(0)
    status_box = st.empty()

    st.subheader("Flux IA (Thinking / Stream)")
    stream_placeholder = st.empty()
    stream_placeholder.code("En attente du démarrage...", language="text")

    def progress_cb(idx: int, total: int, pdf_path: Path) -> None:
        progress_bar.progress(int((idx / max(total, 1)) * 100))
        status_box.info(f"Processing {idx}/{total}: {pdf_path.name}")

    def log_cb(msg: str) -> None:
        logs.append(msg)

    config = AppConfig(
        ollama_url=ollama_url,
        model=model,
        ocr_engine=ocr_engine,
        stream=stream,
        show_thinking=show_thinking,
        guided_zoom=guided_zoom,
        use_page_images_in_review=use_page_images_in_review,
        dpi=int(dpi),
        review_batch_size=int(review_batch_size),
        review_image_max_side=int(review_image_max_side),
        connect_timeout=int(connect_timeout),
        read_timeout=int(read_timeout),
        final_timeout=int(final_timeout),
        qwen_retries=int(qwen_retries),
    )

    stderr_capture = _StderrCapture(stream_placeholder)
    old_stderr = sys.stderr
    sys.stderr = stderr_capture  # type: ignore[assignment]
    try:
        queue = pending_queue if skip_generated else scan_results
        if not queue:
            raise RuntimeError(
                "Queue is empty. Scan PDFs first, or disable 'Skip already generated PDFs'."
            )

        result = run_batch_conversion(
            root_folder=root_path,
            config=config,
            progress_cb=progress_cb,
            log_cb=log_cb,
            pdf_paths=queue,
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr = old_stderr
        stderr_capture.flush()
        st.error(f"Batch execution failed: {exc}")
    else:
        sys.stderr = old_stderr
        stderr_capture.flush()
        progress_bar.progress(100)
        status_box.success(
            f"Done. Success={result.success_count}, Failed={result.failure_count}, "
            f"Global summary={result.global_summary_path}"
        )
        st.subheader("Run Result")
        st.write(
            {
                "total_files": result.total_files,
                "success_count": result.success_count,
                "failure_count": result.failure_count,
                "global_summary_path": str(result.global_summary_path),
            }
        )
        st.subheader("Logs")
        st.code("\n".join(logs) if logs else "No logs captured.", language="text")

st.markdown("---")
st.header("Comparaison PDF ↔ Markdown")

_scan = st.session_state.get("scan_results", [])
if _scan:
    _cmp_pdf_str = st.selectbox(
        "Sélectionner un PDF",
        options=[str(p) for p in _scan],
        index=0,
        key="cmp_select",
    )
else:
    _cmp_pdf_str = st.text_input(
        "Chemin du PDF à comparer (ou lancez d'abord un Scan)",
        value="",
        key="cmp_input",
    )

if _cmp_pdf_str:
    _cmp_pdf = Path(_cmp_pdf_str)
    _ollama_md = _cmp_pdf.with_name(f"{_cmp_pdf.stem}_ollama.md")
    _summary_md = _cmp_pdf.with_name(f"{_cmp_pdf.stem}_summary.md")

    if not _cmp_pdf.exists():
        st.warning(f"Fichier introuvable : {_cmp_pdf}")
    else:
        try:
            import fitz as _fitz  # PyMuPDF
            _doc = _fitz.open(str(_cmp_pdf))
            _total_pages = _doc.page_count
            _doc.close()
        except Exception as _e:
            st.error(f"Impossible d'ouvrir le PDF : {_e}")
            _total_pages = 0

        if _total_pages > 0:
            _page_num = (
                st.slider("Page", min_value=1, max_value=_total_pages, value=1, step=1) - 1
                if _total_pages > 1
                else 0
            )

            _left, _right = st.columns(2)

            with _left:
                st.subheader(f"PDF — page {_page_num + 1} / {_total_pages}")
                try:
                    import fitz as _fitz
                    _doc = _fitz.open(str(_cmp_pdf))
                    _page = _doc[_page_num]
                    _mat = _fitz.Matrix(150 / 72, 150 / 72)
                    _pix = _page.get_pixmap(matrix=_mat)
                    _img_bytes = _pix.tobytes("png")
                    _doc.close()
                    st.image(_img_bytes, width='stretch')
                except Exception as _e:
                    st.error(f"Erreur de rendu : {_e}")

            with _right:
                st.subheader("Markdown généré")
                _tab_conv, _tab_sum = st.tabs(["Conversion (_ollama.md)", "Résumé (_summary.md)"])

                with _tab_conv:
                    if _ollama_md.exists():
                        st.markdown(_ollama_md.read_text(encoding="utf-8"))
                    else:
                        st.info("Fichier `_ollama.md` non trouvé. Lancez d'abord la conversion.")

                with _tab_sum:
                    if _summary_md.exists():
                        st.markdown(_summary_md.read_text(encoding="utf-8"))
                    else:
                        st.info("Fichier `_summary.md` non trouvé. Lancez d'abord la conversion.")

st.markdown("---")
st.caption("Outputs are saved next to source PDFs: <pdf_stem>_ollama.md and <pdf_stem>_summary.md")
