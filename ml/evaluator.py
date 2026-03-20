import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List
from utils.logger import get_logger

logger = get_logger(__name__)

REPORTS_DIR = "outputs/reports"


def generate_report(all_results: List[Dict]) -> str:
    """
    Generates a human-readable evaluation report from all model results.
    Saves as both JSON (for programmatic use) and TXT (for reading).
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # ── JSON Report (full detail) ─────────────────────────
    json_path = os.path.join(REPORTS_DIR, f"evaluation_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"JSON report saved: {json_path}")

    # ── Text Report (human readable) ──────────────────────
    txt_path = os.path.join(REPORTS_DIR, f"evaluation_{timestamp}.txt")
    lines = []
    lines.append("=" * 60)
    lines.append("PHASE 3: ML MODEL EVALUATION REPORT")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append("=" * 60)

    for result in all_results:
        target = result["target"]
        lines.append(f"\n{'─'*60}")
        lines.append(f"TARGET: {target.upper()}")
        lines.append(f"Classes: {result['classes']}")
        lines.append(f"Best Model: {result['best_model']} (F1={result['best_f1']:.4f})")
        lines.append(f"Saved to: {result['model_path']}")
        lines.append(f"\n{'Model Comparison':}")
        lines.append(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10} {'CV F1':>12}")
        lines.append("-" * 70)

        for model_name, metrics in result["model_results"].items():
            lines.append(
                f"{model_name:<25} "
                f"{metrics['accuracy']:>10.4f} "
                f"{metrics['f1_weighted']:>10.4f} "
                f"{str(metrics['roc_auc']):>10} "
                f"{metrics['cv_mean_f1']:>8.4f} ± {metrics['cv_std_f1']:.4f}"
            )

        # Best model's classification report
        best_report = result["model_results"][result["best_model"]]["classification_report"]
        lines.append(f"\nBest Model ({result['best_model']}) — Classification Report:")
        lines.append(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        lines.append("-" * 55)

        for cls, metrics in best_report.items():
            if isinstance(metrics, dict) and "precision" in metrics:
                lines.append(
                    f"{cls:<20} "
                    f"{metrics['precision']:>10.4f} "
                    f"{metrics['recall']:>10.4f} "
                    f"{metrics['f1-score']:>10.4f} "
                    f"{int(metrics['support']):>10}"
                )

        # Confusion matrix
        cm = result["model_results"][result["best_model"]]["confusion_matrix"]
        lines.append(f"\nConfusion Matrix ({result['classes']}):")
        for row in cm:
            lines.append("  " + "  ".join(f"{v:4}" for v in row))

    lines.append(f"\n{'='*60}")
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Text report saved: {txt_path}")

    # Print summary to terminal
    print("\n" + "\n".join(lines))

    return txt_path
