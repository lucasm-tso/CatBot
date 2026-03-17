"""Generation du resume d'incertitude par document."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import AppConfig
from .legacy_bridge import load_legacy_module
from .metrics import round_metrics
from .models import FileConversionResult


def _compute_uncertain_points(metrics: dict[str, Any], status: str) -> list[str]:
    points: list[str] = []
    if status != "success":
        points.append("La conversion du document a echoue.")

    if int(metrics.get("qwen_ocr_timeouts", 0)) > 0:
        points.append("Un ou plusieurs appels OCR ont expire.")
    if int(metrics.get("structured_crop_invalid_schema", 0)) > 0:
        points.append("Le schema JSON OCR structure est invalide au moins une fois.")
    if int(metrics.get("structured_crop_invalid_json", 0)) > 0:
        points.append("La sortie OCR structuree contient un JSON invalide au moins une fois.")
    if int(metrics.get("structured_crop_fallback_plain_ocr", 0)) > 0:
        points.append("Un fallback vers l'OCR brut a ete necessaire sur au moins un zoom.")
    if int(metrics.get("zoom_regions_rejected_small", 0)) > 0:
        points.append("Certaines zones de zoom proposees etaient trop petites et ont ete rejetees.")

    return points


def _uncertainty_score(metrics: dict[str, Any], status: str) -> float:
    if status != "success":
        return 1.0

    score = 0.0
    score += min(0.4, 0.05 * float(metrics.get("qwen_ocr_timeouts", 0)))
    score += min(0.2, 0.03 * float(metrics.get("structured_crop_invalid_schema", 0)))
    score += min(0.2, 0.03 * float(metrics.get("structured_crop_invalid_json", 0)))
    score += min(0.2, 0.02 * float(metrics.get("structured_crop_fallback_plain_ocr", 0)))
    return max(0.0, min(1.0, score))


def _build_ai_summary(
    result: FileConversionResult,
    config: AppConfig,
) -> str:
    legacy = load_legacy_module()
    content = (
        "Produis un resume markdown concis des incertitudes OCR.\n"
        "Concentre-toi uniquement sur les zones potentiellement fausses ou ambigues.\n"
        "Retourne exactement 3 sections markdown: Risques, Points A Verifier En Priorite, Fiabilite.\n"
        "Reste sous 180 mots.\n\n"
        f"Fichier: {result.pdf_path}\n"
        f"Statut: {result.status}\n"
        f"Pages traitees: {result.pages_processed}\n"
        f"Score d'incertitude (0-1): {result.uncertainty_score:.3f}\n"
        f"Points incertains: {result.uncertain_points}\n"
        f"JSON metriques: {json.dumps(result.metrics, ensure_ascii=True)}"
    )
    try:
        return legacy.call_ollama(
            ollama_url=config.ollama_url,
            model=config.resolved_summary_model(),
            content=content,
            timeout_s=config.summary_timeout,
            stream=False,
            show_thinking=False,
            metrics={},
            call_label="uncertainty_summary",
        ).strip()
    except Exception:  # noqa: BLE001
        return "## Risques\nLa generation du resume IA a echoue.\n\n## Points A Verifier En Priorite\n- Verifier manuellement les tableaux numeriques et les zones a faible contraste.\n\n## Fiabilite\nConfiance moderee a faible; s'appuyer sur les metriques de debug ci-dessous."


def write_pdf_summary(result: FileConversionResult, config: AppConfig) -> FileConversionResult:
    """Ecrit le resume markdown par PDF et met a jour les champs d'incertitude."""
    points = _compute_uncertain_points(result.metrics, result.status)
    score = _uncertainty_score(result.metrics, result.status)

    result.uncertain_points = points
    result.uncertainty_score = score

    summary_path = result.pdf_path.with_name(f"{result.pdf_path.stem}_summary.md")
    ai_summary = _build_ai_summary(result=result, config=config)

    lines = [
        f"# Resume de Conversion - {result.pdf_path.name}",
        "",
        "## Metadonnees",
        f"- Genere le: {datetime.now().isoformat(timespec='seconds')}",
        f"- PDF: `{result.pdf_path}`",
        f"- Statut: `{result.status}`",
        f"- Modele: `{result.model}`",
        f"- Pages traitees: {result.pages_processed}",
        f"- Score d'incertitude: {result.uncertainty_score:.3f}",
        "",
        "## Points Incertains",
    ]

    if points:
        lines.extend([f"- {point}" for point in points])
    else:
        lines.append("- Aucun signal critique d'incertitude detecte dans les metriques.")

    if result.error_message:
        lines.extend(["", "## Erreur", f"- {result.error_message}"])

    lines.extend(
        [
            "",
            "## Resume IA",
            ai_summary if ai_summary else "[Resume IA vide]",
            "",
            "## Metriques de Debug",
            "```json",
            json.dumps(
                {
                    "output": str(result.output_md_path) if result.output_md_path else None,
                    "pages_processed": result.pages_processed,
                    "model": result.model,
                    "metrics": round_metrics(result.metrics),
                },
                ensure_ascii=True,
                indent=2,
            ),
            "```",
            "",
        ]
    )

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    result.summary_md_path = summary_path
    return result
