"""
Entry point – 12-factor Factor XII (Admin processes).

Run the full EDA pipeline as a one-shot process:

    python main.py

After the pipeline completes the results are automatically:
  1. Archived to  results/step{N}_{label}/  (next available step number).
  2. Summarised in  big_eda_status.md  (a new section is appended).

Logging is configured here (Factor XI – Logs as event streams) so that all
module loggers write to stdout in a structured, human-readable format.
"""

import logging
import os
import re
import shutil
import sys
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup (Factor XI: treat logs as an event stream to stdout)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
from src.application.eda_pipeline import run_pipeline  # noqa: E402 (after logging setup)
from src.config import settings  # noqa: E402


# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------

def _detect_next_step(results_dir: str) -> int:
    """Return the next step number by scanning existing step* sub-directories.

    Only the leading digit sequence directly after "step" is treated as the
    step number (e.g. ``step11_12_elasticnet`` → 11, ``step1_k100`` → 1).
    This avoids being misled by numbers embedded in the label suffix
    (e.g. "k100", "pca50", "5fold").
    """
    if not os.path.isdir(results_dir):
        return 1
    all_nums: list[int] = []
    for name in os.listdir(results_dir):
        m = re.match(r"step(\d+)", name)
        if m:
            all_nums.append(int(m.group(1)))
    return (max(all_nums) + 1) if all_nums else 1


def _config_label() -> str:
    """Build a short, filesystem-safe label from the current pipeline flags."""
    parts: list[str] = []
    parts.append("cv5" if settings.USE_CV else "single_split")
    parts.append("smote" if settings.USE_SMOTE else "no_smote")
    if settings.REMOVE_LOW_VARIANCE:
        threshold_str = f"{settings.VARIANCE_THRESHOLD:.2f}".replace(".", "p")
        parts.append(f"vt{threshold_str}")
    return "_".join(parts)


def _archive(step_n: int, label: str, results_dir: str, model_names: list[str]) -> str:
    """Copy results/*.csv and results/reductions/*.csv into a new archive dir.

    Only model CSVs whose names appear in *model_names* are copied, so stale
    files from disabled models (e.g. decision_tree) are never carried forward.
    """
    dest = os.path.join(results_dir, f"step{step_n:02d}_{label}")
    dest_red = os.path.join(dest, "reductions")
    os.makedirs(dest_red, exist_ok=True)

    allowed = {f"{m}.csv" for m in model_names}
    for fname in os.listdir(results_dir):
        if fname.endswith(".csv") and fname in allowed:
            shutil.copy2(os.path.join(results_dir, fname), dest)

    src_red = os.path.join(results_dir, "reductions")
    if os.path.isdir(src_red):
        for fname in os.listdir(src_red):
            if fname.endswith(".csv"):
                shutil.copy2(os.path.join(src_red, fname), dest_red)

    logger.info("Results archived → %s", dest)
    return dest


def _summary_table(archive_dir: str) -> str:
    """Build a Markdown table of best-per-model results from an archive folder.

    For each model CSV the function groups by ``dimension_reduction_type``,
    computes mean ± std of ``r2`` across folds, and selects the reduction that
    achieves the highest mean R².  All reductions are shown, sorted descending
    by R² mean.
    """
    _COLS = ("r2", "spearman_r", "within_1_acc", "mae", "qwk")
    model_rows: list[dict] = []

    for fname in sorted(os.listdir(archive_dir)):
        if not fname.endswith(".csv"):
            continue
        model = fname.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(archive_dir, fname))
        except Exception:
            continue
        if df.empty or not all(c in df.columns for c in _COLS):
            continue

        agg = (
            df.groupby("dimension_reduction_type")[list(_COLS)]
            .agg({"r2": ["mean", "std"], "spearman_r": "mean",
                  "within_1_acc": "mean", "mae": "mean", "qwk": "mean"})
        )
        agg.columns = ["r2_mean", "r2_std", "spearman_mean", "within1_mean", "mae_mean", "qwk_mean"]
        agg = agg.sort_values("r2_mean", ascending=False).reset_index()

        for _, row in agg.iterrows():
            std_str = f"±{row['r2_std']:.3f}" if pd.notna(row["r2_std"]) else ""
            model_rows.append({
                "Model": model,
                "Reduction": row["dimension_reduction_type"],
                "R² mean±std": f"{row['r2_mean']:.3f}{std_str}",
                "Spearman ρ": f"{row['spearman_mean']:.3f}",
                "Within±1": f"{row['within1_mean']:.3f}",
                "MAE": f"{row['mae_mean']:.3f}",
                "QWK": f"{row['qwk_mean']:.3f}",
            })

    if not model_rows:
        return "_No results found._"

    headers = ["Model", "Reduction", "R² mean±std", "Spearman ρ", "Within±1", "MAE", "QWK"]
    widths = {
        h: max(len(h), max(len(str(r.get(h, ""))) for r in model_rows))
        for h in headers
    }
    header_line = "| " + " | ".join(h.ljust(widths[h]) for h in headers) + " |"
    sep_line = "| " + " | ".join("-" * widths[h] for h in headers) + " |"
    lines = [header_line, sep_line]
    for r in model_rows:
        lines.append("| " + " | ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers) + " |")
    return "\n".join(lines)


def _append_md(step_n: int, label: str, archive_dir: str, md_path: str) -> None:
    """Append a new section for this run to big_eda_status.md."""
    table = _summary_table(archive_dir)
    ts = datetime.now().strftime("%Y-%m-%d")
    cfg = (
        f"USE_CV={settings.USE_CV}  "
        f"USE_SMOTE={settings.USE_SMOTE}  "
        f"REMOVE_LOW_VARIANCE={settings.REMOVE_LOW_VARIANCE}"
        + (f"  VARIANCE_THRESHOLD={settings.VARIANCE_THRESHOLD}"
           if settings.REMOVE_LOW_VARIANCE else "")
    )
    human_label = label.replace("_", " ").title()
    section = (
        f"\n\n---\n\n"
        f"## Step {step_n} – {human_label}\n\n"
        f"**Date:** {ts}  \n"
        f"**Archive:** `{archive_dir}`  \n"
        f"**Config:** `{cfg}`\n\n"
        f"### Best result per model (mean ± std across CV folds)\n\n"
        f"{table}\n"
    )
    with open(md_path, "a", encoding="utf-8") as fh:
        fh.write(section)
    logger.info("MD updated → %s", md_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the EDA pipeline, archive results, and update the status MD."""
    logger.info("Starting EDA pipeline …")

    results = run_pipeline()

    # Print a compact summary of result counts per model.
    for model_name, records in results.items():
        logger.info("%-20s → %d result records", model_name, len(records))

    # ------------------------------------------------------------------
    # Auto-archive + MD update
    # ------------------------------------------------------------------
    step_n = _detect_next_step(settings.RESULTS_DIR)
    label = _config_label()
    archive_dir = _archive(step_n, label, settings.RESULTS_DIR, list(results.keys()))
    _append_md(step_n, label, archive_dir, "docs/big_eda_status.md")

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
