import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
LABEL_INFER = {
    "azure": REPO_ROOT / "label" / "processor" / "AzureOpenAI.py",
    "vllm": REPO_ROOT / "label" / "processor" / "vLLM.py",
}
FEATURE_INFER = {
    "azure": REPO_ROOT / "feature" / "processor" / "AzureOpenAI.py",
    "vllm": REPO_ROOT / "feature" / "processor" / "vLLM.py",
}
LABEL_EVAL = REPO_ROOT / "label" / "processor" / "eval.py"
FEATURE_EVAL = REPO_ROOT / "feature" / "processor" / "eval.py"
POSITIVE_VALUE = 1  # feature pipeline expects 1 for true positives
CXR_LABEL_COLUMNS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


@dataclass
class DatasetSpec:
    tag: str                 # “generated”, “reference”, etc.
    reports: Path            # CSV with columns: study_id, report
    label_gt: Path | None = None    # CSV for label evaluation and TP filtering
    feature_gt: Path | None = None  # JSON (or CSV) for feature evaluation


async def run_cmd(tag: str, command: list[str]) -> None:
    print(f"[{tag}] {' '.join(command)}")
    proc = await asyncio.create_subprocess_exec(*command, cwd=str(REPO_ROOT))
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"{tag} failed with exit code {rc}")


async def run_label_inference(spec: DatasetSpec, backbone: str, model: str, output_root: Path) -> tuple[Path, Path]:
    out_dir = output_root / spec.tag / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tmp").mkdir(exist_ok=True)

    cmd = [
        sys.executable,
        str(LABEL_INFER[backbone]),
        "--model_name",
        model,
        "--reports",
        str(spec.reports),
        "--output",
        str(out_dir),
    ]
    await run_cmd(f"{spec.tag}-label-infer", cmd)
    pred_json = out_dir / "tmp" / f"output_labels_{model}.json"
    return out_dir, pred_json


async def run_label_evaluation(spec: DatasetSpec, out_dir: Path, model: str) -> None:
    if spec.label_gt is None:
        return
    cmd = [
        sys.executable,
        str(LABEL_EVAL),
        "--gt_dir",
        str(spec.label_gt),
        "--gen_dir",
        str(out_dir),
        "--model_name",
        model,
    ]
    await run_cmd(f"{spec.tag}-label-eval", cmd)


def normalize_label_value(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "positive", "true"}:
        return 1
    if text in {"0", "negative", "false"}:
        return 0
    if text in {"-1", "unclear", "n/a", "nan"}:
        return -1
    return -1


def label_json_to_csv(json_path: Path, output_csv: Path) -> Path:
    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)

    rows: list[dict[str, object]] = []
    for study_id, conditions in data.items():
        row = {"study_id": study_id}
        if isinstance(conditions, str):
            try:
                conditions = json.loads(conditions)
            except json.JSONDecodeError:
                conditions = {}
        if not isinstance(conditions, dict):
            conditions = {}
        for label in CXR_LABEL_COLUMNS:
            row[label] = normalize_label_value(conditions.get(label))
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="study_id")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def prepare_feature_label_csv(
    gt_csv: Path,
    pred_json: Path,
    output_csv: Path,
) -> Path:
    with open(pred_json, encoding="utf-8") as fh:
        preds = json.load(fh)

    df_gt = pd.read_csv(gt_csv, dtype={'study_id': str}).set_index("study_id")
    df_tp = pd.DataFrame(0, index=df_gt.index, columns=df_gt.columns)

    for study_id, conditions in preds.items():
        if study_id not in df_tp.index:
            continue
        for condition, value in conditions.items():
            if condition not in df_tp.columns:
                continue
            if str(value).lower() == "positive" and df_gt.loc[study_id, condition] == 1:
                df_tp.loc[study_id, condition] = POSITIVE_VALUE

    df_tp.insert(0, "study_id", df_tp.index)
    df_tp.to_csv(output_csv, index=False)
    return output_csv


async def run_feature_inference(
    spec: DatasetSpec,
    backbone: str,
    model: str,
    filtered_labels: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    out_dir = output_root / spec.tag / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tmp").mkdir(exist_ok=True)

    cmd = [
        sys.executable,
        str(FEATURE_INFER[backbone]),
        "--model",
        model,
        "--reports",
        str(spec.reports),
        "--labels",
        str(filtered_labels),
        "--output",
        str(out_dir),
    ]
    await run_cmd(f"{spec.tag}-feature-infer", cmd)
    gen_json = out_dir / "tmp" / f"output_feature_{model}.json"
    return out_dir, gen_json


async def run_feature_evaluation(
    spec: DatasetSpec,
    out_dir: Path,
    gen_json: Path,
    model: str,
    enable_llm_metric: bool,
    scoring_llm: str | None,
) -> None:
    if spec.feature_gt is None:
        return
    cmd = [
        sys.executable,
        str(FEATURE_EVAL),
        "--gen_path",
        str(gen_json),
        "--gt_path",
        str(spec.feature_gt),
        "--metric_path",
        str(out_dir),
        "--model_name",
        model,
    ]
    if enable_llm_metric:
        cmd.append("--enable_llm_metric")
        if scoring_llm:
            cmd.extend(["--scoring_llm", scoring_llm])
    await run_cmd(f"{spec.tag}-feature-eval", cmd)


async def orchestrate(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    specs: list[DatasetSpec] = [DatasetSpec("generated", args.gen_reports.resolve())]
    if args.gt_reports:
        specs.append(DatasetSpec("reference", args.gt_reports.resolve()))


    # Stage 1: label inference
    label_results = []
    for spec in specs:
        label_results.append(
            await run_label_inference(spec, args.label_backbone, args.label_model, output_root)
        )

    label_dirs = {spec.tag: result[0] for spec, result in zip(specs, label_results)}
    pred_jsons = {spec.tag: result[1] for spec, result in zip(specs, label_results)}

    generated_spec = specs[0]
    reference_spec = next((spec for spec in specs if spec.tag == "reference"), None)
    if reference_spec is not None and "reference" in pred_jsons:
        gt_csv_path = label_dirs["reference"] / f"output_labels_{args.label_model}.csv"
        label_json_to_csv(pred_jsons["reference"], gt_csv_path)
        generated_spec.label_gt = gt_csv_path
        reference_spec.label_gt = gt_csv_path

    # Stage 2: label evaluation (gt and gen input as a pair)
    await run_label_evaluation(generated_spec, label_dirs["generated"], args.label_model)

    # Stage 3: build filtered label CSV for only true positive conditions in gt and gen
    filtered_csvs: dict[str, Path] = {}
    if generated_spec.label_gt is not None:
        filtered_csvs["generated"] = await asyncio.to_thread(
            prepare_feature_label_csv,
            generated_spec.label_gt,
            pred_jsons["generated"],
            output_root / "generated" / f"filtered_tp_labels_{args.label_model}.csv",
        )

    # Stage 4: feature inference (gt and gen)
    feature_specs = [spec for spec in specs]
    feature_dirs: dict[str, Path] = {}
    feature_jsons: dict[str, Path] = {}
    if feature_specs:
        feature_results = []
        for spec in feature_specs:
            feature_results.append(
                await run_feature_inference(
                    spec,
                    args.feature_backbone,
                    args.feature_model,
                    filtered_csvs["generated"],
                    output_root,
                )
            )
        feature_dirs = {spec.tag: result[0] for spec, result in zip(feature_specs, feature_results)}
        feature_jsons = {spec.tag: result[1] for spec, result in zip(feature_specs, feature_results)}

    # Stage 5: feature evaluation (gt and gen input as a pair)
    if "reference" in feature_jsons:
        specs[0].feature_gt = feature_jsons["reference"]
    if "generated" in feature_jsons:
        await run_feature_evaluation(
            specs[0],
            feature_dirs["generated"],
            feature_jsons["generated"],
            args.feature_model,
            args.feature_eval_enable_llm,
            args.feature_eval_scoring_llm,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CLEAR label and feature pipelines with stage barriers.")
    parser.add_argument("--gen-reports", type=Path, required=True, help="CSV of generated reports.")
    parser.add_argument("--gt-reports", type=Path, help="CSV of reference reports.")

    parser.add_argument("--label-backbone", choices=("azure", "vllm"), default="vllm")
    parser.add_argument("--label-model", required=True)
    parser.add_argument("--feature-backbone", choices=("azure", "vllm"), default="vllm")
    parser.add_argument("--feature-model", required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs", help="Directory to store outputs.")
    parser.add_argument("--feature-eval-enable-llm", action="store_true", help="Enable LLM-based IE scoring during feature evaluation.")
    parser.add_argument("--feature-eval-scoring-llm", type=str, help="Model name for LLM-based IE scoring.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(orchestrate(args))
    print('Successfully finished all evaluation!')


if __name__ == "__main__":
    main()
