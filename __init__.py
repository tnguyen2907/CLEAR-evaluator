import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def compute_clear(
    refs: List[str],
    hyps: List[str],
    debug_dir: Optional[str] = None,
    label_model: str = "llama-3.1-70b-instruct",
    feature_model: str = "llama-3.1-8b-instruct",
) -> Dict[str, object]:
    """Run the full CLEAR evaluation pipeline and return per-report scores.

    Args:
        refs: List of reference radiology reports.
        hyps: List of generated (hypothesis) radiology reports.
        debug_dir: If provided, intermediate files are written here and kept.
        label_model: Model name for label extraction (must exist in label/configs/models.py).
        feature_model: Model name for feature extraction (must exist in feature/configs/models.py).

    Returns:
        Dictionary with per-report metric lists and raw label/feature dicts,
        all in the same order as the input lists.
    """
    if debug_dir:
        td = Path(debug_dir).resolve()
        td.mkdir(parents=True, exist_ok=True)
    else:
        td = Path(tempfile.mkdtemp())

    gen_csv = td / "gen.csv"
    gt_csv = td / "gt.csv"
    out_dir = td / "runs"
    out_dir.mkdir(exist_ok=True)

    ids = ["s" + str(i).zfill(9) for i in range(len(hyps))]
    pd.DataFrame({"study_id": ids, "report": hyps}).to_csv(gen_csv, index=False)
    pd.DataFrame({"study_id": ids, "report": refs}).to_csv(gt_csv, index=False)

    cmd = [
        sys.executable, "-m", "clear_evaluator.main",
        "--gen-reports", str(gen_csv),
        "--gt-reports", str(gt_csv),
        "--label-backbone", "vllm",
        "--label-model", label_model,
        "--feature-backbone", "vllm",
        "--feature-model", feature_model,
        "--output-root", str(out_dir),
    ]

    try:
        subprocess.run(cmd, check=True)

        label_metrics_path = out_dir / "generated" / "labels" / f"label_metrics_per_report_{label_model}.csv"
        qa_metrics_path    = out_dir / "generated" / "features" / f"results_qa_per_report_{feature_model}.csv"
        ie_metrics_path    = out_dir / "generated" / "features" / f"results_ie_per_report_{feature_model}.csv"

        label_metrics_df = pd.read_csv(label_metrics_path, dtype={"study_id": str})
        qa_metrics_df    = pd.read_csv(qa_metrics_path, dtype={"study_id": str})
        ie_metrics_df    = pd.read_csv(ie_metrics_path, dtype={"study_id": str})

        gen_label_json_path   = out_dir / "generated" / "labels"   / "tmp" / f"output_labels_{label_model}.json"
        gt_label_json_path    = out_dir / "reference" / "labels"   / "tmp" / f"output_labels_{label_model}.json"
        gen_feature_json_path = out_dir / "generated" / "features" / "tmp" / f"output_feature_{feature_model}.json"
        gt_feature_json_path  = out_dir / "reference" / "features" / "tmp" / f"output_feature_{feature_model}.json"

        with open(gen_label_json_path, encoding="utf-8") as j:
            gen_label_json = json.load(j)
        with open(gt_label_json_path, encoding="utf-8") as j:
            gt_label_json = json.load(j)
        with open(gen_feature_json_path, encoding="utf-8") as j:
            gen_feature_json = json.load(j)
        with open(gt_feature_json_path, encoding="utf-8") as j:
            gt_feature_json = json.load(j)

        assert (
            ids
            == label_metrics_df["study_id"].tolist()
            == qa_metrics_df["study_id"].tolist()
            == ie_metrics_df["study_id"].tolist()
            == list(gen_label_json.keys())
            == list(gen_feature_json.keys())
            == list(gt_label_json.keys())
            == list(gt_feature_json.keys())
        ), "Wrong id order"

        results = {
            "label_presence": label_metrics_df["pos_f1_per_report"].tolist(),
            "first_occurence": qa_metrics_df["First Occurrence"].tolist(),
            "change": qa_metrics_df["Change"].tolist(),
            "severity": qa_metrics_df["Severity"].tolist(),
            "location": ie_metrics_df["Descriptive Location"].tolist(),
            "recommendation": ie_metrics_df["Recommendation"].tolist(),
            "gt_labels": list(gt_label_json.values()),
            "gen_labels": list(gen_label_json.values()),
            "gt_features": list(gt_feature_json.values()),
            "gen_features": list(gen_feature_json.values()),
        }
        return results
    finally:
        if not debug_dir:
            shutil.rmtree(td)
