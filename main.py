import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
LABEL_INFER = {
    "azure": "clear_evaluator.label.processor.AzureOpenAI",
    "vllm": "clear_evaluator.label.processor.vLLM",
}
FEATURE_INFER = {
    "azure": "clear_evaluator.feature.processor.AzureOpenAI",
    "vllm": "clear_evaluator.feature.processor.vLLM",
}
LABEL_EVAL = "clear_evaluator.label.processor.eval"
FEATURE_EVAL = "clear_evaluator.feature.processor.eval"
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
    tag: str                 # "generated", "reference", etc.
    reports: Path            # CSV with columns: study_id, report
    label_gt: Path | None = None    # CSV for label evaluation and TP filtering
    feature_gt: Path | None = None  # JSON (or CSV) for feature evaluation


def get_vllm_config(backbone: str, model: str, stage: str):
    """Import the model config to get TP/DP for torchrun launcher."""
    if backbone != "vllm":
        return None
    if stage == "label":
        from label.configs.models import MODEL_CONFIGS
    else:
        from feature.configs.models import MODEL_CONFIGS
    return MODEL_CONFIGS.get(model)


async def run_cmd(tag: str, command: list[str]) -> None:
    print(f"[{tag}] {' '.join(command)}")
    proc = await asyncio.create_subprocess_exec(*command)
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"{tag} failed with exit code {rc}")


def build_torchrun_cmd(module: str, nproc: int, args: list[str]) -> list[str]:
    """Build a torchrun command for launching vLLM with data parallelism."""
    return [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc-per-node={nproc}",
        "-m", module,
    ] + args


def build_inference_cmd(backbone: str, module: str, model: str, extra_args: list[str], stage: str) -> list[str]:
    """Build the subprocess command, using torchrun for vLLM backends."""
    if backbone == "vllm":
        config = get_vllm_config(backbone, model, stage)
        tp = config.get("tensor_parallel_size", 1)
        dp = config.get("data_parallel_size", 1)
        nproc = tp * dp
        return build_torchrun_cmd(module, nproc, extra_args)
    else:
        return [sys.executable, "-m", module] + extra_args


async def run_label_inference(spec: DatasetSpec, backbone: str, model: str, output_root: Path) -> tuple[Path, Path]:
    out_dir = output_root / spec.tag / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tmp").mkdir(exist_ok=True)

    extra_args = [
        "--model_name", model,
        "--reports", str(spec.reports),
        "--output", str(out_dir),
    ]
    cmd = build_inference_cmd(backbone, LABEL_INFER[backbone], model, extra_args, "label")
    await run_cmd(f"{spec.tag}-label-infer", cmd)
    pred_json = out_dir / "tmp" / f"output_labels_{model}.json"
    return out_dir, pred_json


async def run_label_evaluation(spec: DatasetSpec, out_dir: Path, model: str) -> None:
    if spec.label_gt is None:
        return
    cmd = [
        sys.executable,
        "-m", LABEL_EVAL,
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
        if isinstance(conditions, str):
            try:
                conditions = json.loads(conditions)
            except json.JSONDecodeError:
                continue
        if not isinstance(conditions, dict):
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

    extra_args = [
        "--model", model,
        "--reports", str(spec.reports),
        "--labels", str(filtered_labels),
        "--output", str(out_dir),
    ]
    cmd = build_inference_cmd(backbone, FEATURE_INFER[backbone], model, extra_args, "feature")
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
        "-m", FEATURE_EVAL,
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


def merge_report_csvs(csv_paths: dict[str, Path], output_csv: Path) -> Path:
    """Merge multiple report CSVs into one, prefixing study_id with source tag."""
    dfs = []
    for tag, path in csv_paths.items():
        df = pd.read_csv(path, dtype={'study_id': str})
        df['study_id'] = tag + '/' + df['study_id']
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return output_csv


def split_label_json(json_path: Path, tags: list[str], output_dir: Path, model: str) -> dict[str, Path]:
    """Split a merged label JSON back into per-source JSONs."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    results = {tag: {} for tag in tags}
    for study_id, value in data.items():
        # study_id is "tag/orig_study_id"
        tag, orig_id = study_id.split('/', 1)
        if tag in results:
            results[tag][orig_id] = value

    paths = {}
    for tag, tag_data in results.items():
        out_path = output_dir / tag / "labels" / "tmp" / f"output_labels_{model}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(tag_data, f, ensure_ascii=False, indent=4)
        paths[tag] = out_path
    return paths


def merge_label_csvs(csv_paths: dict[str, Path], output_csv: Path) -> Path:
    """Merge multiple label CSVs into one, prefixing study_id with source tag."""
    dfs = []
    for tag, path in csv_paths.items():
        df = pd.read_csv(path, dtype={'study_id': str})
        df['study_id'] = tag + '/' + df['study_id']
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return output_csv


def split_feature_json(json_path: Path, tags: list[str], output_dir: Path, model: str) -> dict[str, Path]:
    """Split a merged feature JSON back into per-source JSONs."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    results = {tag: {} for tag in tags}
    for study_id, value in data.items():
        tag, orig_id = study_id.split('/', 1)
        if tag in results:
            results[tag][orig_id] = value

    paths = {}
    for tag, tag_data in results.items():
        out_path = output_dir / tag / "features" / "tmp" / f"output_feature_{model}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(tag_data, f, ensure_ascii=False, indent=4)
        paths[tag] = out_path
    return paths


async def orchestrate(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    specs: list[DatasetSpec] = [DatasetSpec("generated", args.gen_reports.resolve())]
    if args.gt_reports:
        specs.append(DatasetSpec("reference", args.gt_reports.resolve()))

    tags = [spec.tag for spec in specs]

    # ── Stage 1: label inference (merged gen+ref → single model run) ──
    merged_reports_csv = output_root / "tmp" / "merged_reports.csv"
    merge_report_csvs({spec.tag: spec.reports for spec in specs}, merged_reports_csv)

    # Create a temporary spec for the merged run
    merged_label_spec = DatasetSpec("merged", merged_reports_csv)
    label_out_dir = output_root / "merged" / "labels"
    label_out_dir.mkdir(parents=True, exist_ok=True)
    (label_out_dir / "tmp").mkdir(exist_ok=True)

    extra_args = [
        "--model_name", args.label_model,
        "--reports", str(merged_reports_csv),
        "--output", str(label_out_dir),
    ]
    cmd = build_inference_cmd(args.label_backbone, LABEL_INFER[args.label_backbone], args.label_model, extra_args, "label")
    await run_cmd("merged-label-infer", cmd)

    # Split merged results back to per-source
    merged_label_json = label_out_dir / "tmp" / f"output_labels_{args.label_model}.json"
    pred_jsons = split_label_json(merged_label_json, tags, output_root, args.label_model)

    # Also create per-source label output dirs for eval
    label_dirs = {}
    for spec in specs:
        d = output_root / spec.tag / "labels"
        d.mkdir(parents=True, exist_ok=True)
        label_dirs[spec.tag] = d

    generated_spec = specs[0]
    reference_spec = next((spec for spec in specs if spec.tag == "reference"), None)
    if reference_spec is not None and "reference" in pred_jsons:
        gt_csv_path = label_dirs["reference"] / f"output_labels_{args.label_model}.csv"
        label_json_to_csv(pred_jsons["reference"], gt_csv_path)
        generated_spec.label_gt = gt_csv_path
        reference_spec.label_gt = gt_csv_path

    # ── Stage 2: label evaluation ──
    await run_label_evaluation(generated_spec, label_dirs["generated"], args.label_model)

    # ── Stage 3: build filtered label CSV ──
    filtered_csvs: dict[str, Path] = {}
    if generated_spec.label_gt is not None:
        filtered_csvs["generated"] = await asyncio.to_thread(
            prepare_feature_label_csv,
            generated_spec.label_gt,
            pred_jsons["generated"],
            output_root / "generated" / f"filtered_tp_labels_{args.label_model}.csv",
        )

    # Wait for vLLM CUDA cleanup before loading the next model
    print("Waiting 15s for GPU memory cleanup...")
    time.sleep(15)

    # ── Stage 4: feature inference (merged gen+ref → single model run) ──
    feature_specs = [spec for spec in specs]
    feature_dirs: dict[str, Path] = {}
    feature_jsons: dict[str, Path] = {}
    if feature_specs and "generated" in filtered_csvs:
        # Merge reports and labels for all sources
        merged_feature_reports = output_root / "tmp" / "merged_feature_reports.csv"
        merge_report_csvs({spec.tag: spec.reports for spec in feature_specs}, merged_feature_reports)

        # For feature labels, we use the same filtered TP labels for all sources
        # but need to prefix study_ids to match merged reports
        merged_feature_labels = output_root / "tmp" / "merged_feature_labels.csv"
        label_csv_paths = {spec.tag: filtered_csvs["generated"] for spec in feature_specs}
        merge_label_csvs(label_csv_paths, merged_feature_labels)

        feature_out_dir = output_root / "merged" / "features"
        feature_out_dir.mkdir(parents=True, exist_ok=True)
        (feature_out_dir / "tmp").mkdir(exist_ok=True)

        extra_args = [
            "--model", args.feature_model,
            "--reports", str(merged_feature_reports),
            "--labels", str(merged_feature_labels),
            "--output", str(feature_out_dir),
        ]
        cmd = build_inference_cmd(args.feature_backbone, FEATURE_INFER[args.feature_backbone], args.feature_model, extra_args, "feature")
        await run_cmd("merged-feature-infer", cmd)

        merged_feature_json = feature_out_dir / "tmp" / f"output_feature_{args.feature_model}.json"
        split_jsons = split_feature_json(merged_feature_json, tags, output_root, args.feature_model)

        for spec in feature_specs:
            d = output_root / spec.tag / "features"
            d.mkdir(parents=True, exist_ok=True)
            feature_dirs[spec.tag] = d
            if spec.tag in split_jsons:
                feature_jsons[spec.tag] = split_jsons[spec.tag]

    # ── Stage 5: feature evaluation ──
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
