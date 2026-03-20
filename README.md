# CLEAR: A Clinically Grounded Tabular Framework for Radiology Report Evaluation

CLEAR provides an end-to-end evaluator for radiology reports built on the taxonomy introduced in [CLEAR (EMNLP Findings 2025)](https://arxiv.org/abs/2505.16325). The pipeline pairs label-level reasoning with fine-grained feature extraction so you can score generated reports against radiologist-grade annotations.

![CLEAR overview](pics/CLEAR_overview.png)

## Highlights
- Covers both condition classification and detailed feature extraction with consistent schema enforcement.
- Supports open-source (vLLM) and closed-source (Azure OpenAI) backends via declarative model configs.
- Ships with orchestration scripts that stage inference, evaluation, and intermediate data hand-offs.
- Produces granular metrics (per-condition F1, QA/IE scores, optional LLM-based scoring) for auditability.

## Additional Resources
1. **CLEAR-Bench** (coming soon): our expert evaluation dataset, to be released on [PhysioNet](https://physionet.org/).
2. **vLLM**: see the [official documentation](https://docs.vllm.ai/en/latest/) for deployment and performance tuning.
3. **Responsible AI Use**: follow the [Responsible Use of MIMIC Data with Online Services like GPT](https://physionet.org/news/post/gpt-responsible-use) guidelines. We recommend the [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-foundry/models/openai/) for secure commercial model access.

## Repository Layout
```
.
├── environment.yaml           # Conda environment definition
├── feature/                   # Feature extraction prompts, configs, processors
├── label/                     # Label extraction prompts, configs, processors
├── main.py                    # Orchestrates label + feature pipelines end-to-end
├── models/                    # Instructions for fine-tuning bespoke evaluators
├── run.bash                   # Convenience wrapper around main.py
├── data/                      # (User-provided) report and label CSVs
├── runs/                      # Default output directory created at runtime
└── README.md
```

## Setup

### Prerequisites
- Linux environment with Python 3.10+ (matching `environment.yaml`).
- Conda (recommended) or an equivalent virtual environment manager.
- GPU with CUDA drivers when running vLLM backends.
- Azure OpenAI subscription when using the Azure processors.

### Install
```bash
cd CLEAR-evaluator
conda env create -f environment.yaml
conda activate clear-evaluator
```

## Data Requirements
- **Generated reports (`--gen-reports`)**: CSV with at least `study_id` and `report` columns. The `report` field should include both FINDINGS and IMPRESSION sections.
- **Reference reports (`--gt-reports`, optional)**: CSV with the same schema as generated reports. When supplied, the pipeline will derive ground-truth labels and features from the reference run.

## Configuring Models
Model definitions live in `label/configs/models.py` and `feature/configs/models.py`.
- **Azure entries** must include `api_key`, `api_version`, `endpoint`, and `deployment`. Optional fields such as `max_tokens` can be added per deployment.
- **vLLM entries** must include `model_path`, `temperature`, `max_tokens`, and `tensor_parallel_size`. Ensure the model weights are accessible on disk and compatible with your hardware.

Prompts for each stage are defined in the paired `prompts.py` files. You can extend or adjust them to suit new conditions or features.

## Running the Evaluator
`run.bash` orchestrates the entire pipeline. Open the script and edit the configuration block at the top to point to your data, models, and preferred output directory before running it.

Key variables inside `run.bash`:
- `GEN_REPORTS` / `GT_REPORTS`: CSVs containing generated and (optional) reference reports with `study_id` and `report` columns. Leave `GT_REPORTS` empty if you do not have references.
- `LABEL_BACKBONE` / `FEATURE_BACKBONE`: choose `azure` or `vllm` for each stage.
- `LABEL_MODEL` / `FEATURE_MODEL`: model identifiers defined in `label/configs/models.py` and `feature/configs/models.py`.
- `OUTPUT_ROOT`: directory where the pipeline writes outputs (`runs/<timestamp>` by default).
- `ENABLE_LLM` and `SCORING_LLM`: toggle the optional LLM-based IE metrics and choose the scoring model.
- `PYTHON_BIN`: interpreter used to run `main.py` (defaults to the active environment).

After updating those values, launch the pipeline with:
```bash
bash run.bash
```

### Stage Outputs
Each run builds `runs/<timestamp>/` with the following structure:
- `generated/labels/tmp/output_labels_<MODEL>.json`: raw label predictions.
- `generated/output_labels_<MODEL>.csv`: normalized label table used for evaluation.
- `generated/filtered_tp_labels_<MODEL>.csv`: positive-condition filter passed to the feature stage.
- `generated/features/tmp/output_feature_<MODEL>.json`: extracted feature set.
- `generated/features/results_qa_avg_<MODEL>.csv`, `results_ie_avg_<MODEL>.csv`: quantitative metrics per feature type.
- `generated/label_metrics_<MODEL>.csv`: label evaluation summary.

When a reference dataset is provided, the same sub-directories are created under `reference/` for comparison and for deriving ground-truth annotations.

## Module Scripts
- `label/run_label.bash` and `feature/run_feature.bash` show minimal examples for invoking processors in isolation.
- `label/processor/eval.py` reports per-condition positive/negative F1 scores, including `Pos F1`, `Pos F1_5`, `Neg F1`, and micro variants.
- `feature/processor/eval.py` reports QA metrics (`Acc. micro/macro`, `F1 micro/macro`) plus IE metrics (`o1-mini score`, `ROUGE-L`, `BLEU-4`). Pass `--enable_llm_metric` and `--scoring_llm` to compute the LLM-based IE score.

## Tips & Troubleshooting
- Ensure vLLM model definitions specify `temperature`, `max_tokens`, and `tensor_parallel_size`; missing fields will trigger runtime errors.
- When using Azure, double-check that environment keys match your active subscription and that the deployment name aligns with the configured model.
- Reports must contain a `report` column with combined FINDINGS and IMPRESSION text; missing sections degrade model performance.
- CLEAR assumes the label schema `{0: negative, 1: positive, -1: unclear}`; normalize upstream data before ingestion to avoid misaligned metrics.

## Citation
If you use CLEAR in academic work, please cite the original CLEAR paper linked above.

## Additions Relative to the Original Baseline Code
Compared with the original baseline code state, this repository adds several practical changes to make the evaluator easier to run, easier to integrate, and more robust in downstream evaluation workflows.

### 1. Package Entry Point
This repo has been cleaned up so it can be installed and used as a Python package rather than only as a collection of scripts. In particular:

- `__init__.py` now exposes `compute_clear(...)` as a package-level entry point.
- Package/import cleanup has been applied across modules so the code can be invoked through `clear_evaluator.*` imports and standard packaging metadata in `pyproject.toml`.
- `compute_clear(...)` runs the pipeline on temporary CSVs and asserts that intermediate outputs preserve the exact input `study_id` order.

### 2. Per-Report Scoring Outputs
The evaluation code now includes per-report outputs in addition to aggregate metrics:

- `label/processor/eval.py` writes `label_metrics_per_report_<model>.csv`, including per-report positive F1.
- `feature/processor/eval.py` writes `results_qa_per_report_<model>.csv` and `results_ie_per_report_<model>.csv`.
- QA per-report scores are computed as mean row accuracies by `study_id`.
- IE per-report scores currently use ROUGE-based row means by `study_id`.

### 3. Refactored vLLM Data-Parallel Workflow
The vLLM execution path has been refactored so data parallelism is handled at the application level by the pipeline orchestration:

- `main.py` now merges generated and reference inputs within each stage, shards the merged CSV across replicas, assigns one subprocess per GPU group, and splits outputs back before evaluation.
- `data_parallel_size` remains in model configs, but it is used by `main.py` as an orchestration parameter that determines how many replica groups to launch.
- vLLM execution uses the standard `LLM(... tensor_parallel_size=...)` path while keeping the merged-per-stage workflow for label and feature inference.

### 4. Robustness and Behavioral Fixes
Several code paths were also adjusted to preserve report coverage and make the evaluator more fault-tolerant:

- `label/processor/vLLM.py` preserves malformed label outputs under the same `study_id` instead of crashing or dropping them.
- `feature/processor/vLLM.py` batches feature prompts in a single `llm.generate(...)` call per replica.
- Reports with no positive conditions are preserved as empty dicts rather than being lost.
- Malformed label JSON rows default to all `-1` during evaluation so the report is still included in scoring.
- The per-report feature scoring path is intentionally lightweight; it is sufficient for current usage, though it may need additional cleanup for full compatibility with the optional LLM-based IE scoring path.
