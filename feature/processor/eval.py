# numpy version must be lower than 2.0.0 to adapt nltk
import json
import re
import pandas as pd
import numpy as np
import math
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from typing import Optional, Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from openai import AzureOpenAI
from tqdm import tqdm

from configs.prompts import LLMMetricPrompts
from configs.models import MODEL_CONFIGS

# Feature configuration
FEATURE_CONFIG = {
    "IE": ['Descriptive Location', "Recommendation"],
    "QA": ['First Occurrence', 'Change', 'Severity'],
    "Support Devices": ['Descriptive Location', 'Recommendation'],
    "All": ['First Occurrence', 'Change', 'Severity', 'Descriptive Location', 'Recommendation']
}


def load_json_file(path: str, description: str) -> dict:
    """Load a JSON file with a helpful error message if it fails."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found at {path}")

    with open(path, "r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse {description} at {path}: {exc}") from exc


def load_text_file(path: str, description: str) -> str:
    """Load a UTF-8 text file and return its content."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found at {path}")

    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def extract_and_parse_json(text):
    """
    Extract and parse JSON arrays from text or provide reasonable defaults
    
    This function tries multiple approaches to parse JSON from potentially malformed strings:
    1. Direct parsing
    2. Extracting bracket-enclosed content
    3. Looking for conclusion sections
    4. Handling special values
    5. Extracting quoted items
    """
    # 1. Try parsing the entire text directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 2. Try to find and extract JSON arrays
    try:
        # Look for the last bracket-enclosed content
        array_pattern = r'\[(.*?)\](?=[^[\]]*$)'
        match = re.search(array_pattern, text, re.DOTALL)
        if match:
            content = match.group(0)
            # Replace single quotes with double quotes
            content = content.replace("'", '"')
            return json.loads(content)
    except:
        pass
    
    # 3. Try to extract final answer list from longer LLM output
    try:
        # Look for common conclusion markers
        conclusion_markers = ["Therefore", "In conclusion", "So", "Finally", "Hence"]
        for marker in conclusion_markers:
            if marker in text:
                conclusion_part = text.split(marker)[-1].strip()
                # Find bracket content
                array_match = re.search(r'\[(.*?)\]', conclusion_part)
                if array_match:
                    content = array_match.group(0)
                    # Process content to make it valid JSON
                    content = content.replace("'", '"')
                    if '"' not in content:  # Add quotes if missing
                        content = content.replace("[", '["').replace("]", '"]').replace(", ", '", "')
                    return json.loads(content)
    except:
        pass
    
    # 4. Handle special values like "nan" or "NaN"
    if text.strip().lower() in ["nan", "\"nan\"", "'nan'"]:
        return ["nan"]
    
    # 5. Try to extract potential list items from the text
    try:
        # Find all quoted content
        items = re.findall(r'["\'](.*?)["\']', text)
        if items:
            return items
    except:
        pass
    
    # 6. If all attempts fail, return a list containing the original text or default value
    print(f"Unable to parse JSON: {text[:50]}...")
    if text:
        # If text is not empty, return the last sentence or first 50 characters as an element
        last_sentence = text.split(".")[-2].strip() if "." in text else text[:50].strip()
        return [last_sentence]
    else:
        # If text is empty, return default value
        return ["unclear"]
    


def convert_feature_df(dict_gt: dict, 
                       dict_gen: dict, 
                       name: str,
                       mode: str) -> pd.DataFrame:
    """
    Convert feature dictionaries to DataFrame format for evaluation
    """
    temp_data = []
     
    for id, gt_condition in dict_gt.items():
        if id not in dict_gen:
            raise KeyError(f"Missing generated features for study_id '{id}'")

        gen_condition = dict_gen[id]

        missing_conditions = set(gt_condition) - set(gen_condition)
        extra_conditions = set(gen_condition) - set(gt_condition)
        if missing_conditions or extra_conditions:
            raise ValueError(
                f"Condition mismatch for study_id '{id}'. Missing in generated: {sorted(missing_conditions)}; "
                f"unexpected in generated: {sorted(extra_conditions)}"
            )
        
        appended = False
        
        for condition, gt_feature_dict in gt_condition.items():
            if condition == 'Support Devices' and name not in FEATURE_CONFIG["Support Devices"]:
                continue

            if condition not in gen_condition:
                raise KeyError(f"Generated features missing condition '{condition}' for study_id '{id}'")

            gen_feature_dict = gen_condition[condition]

            if name not in gt_feature_dict:
                raise KeyError(f"Ground truth missing feature '{name}' for condition '{condition}' and study_id '{id}'")
            if name not in gen_feature_dict:
                raise KeyError(f"Generated output missing feature '{name}' for condition '{condition}' and study_id '{id}'")

            gt_answer = gt_feature_dict[name]
            gen_answer = gen_feature_dict[name]

            # Use the improved JSON extraction function
            gt_feature = extract_and_parse_json(gt_answer)
            gen_feature = extract_and_parse_json(gen_answer)
                
            assert isinstance(gt_feature, list), f"gt {id} {condition} {name} exists incompatible format"
            assert isinstance(gen_feature, list), f"gen {id} {condition} {name} exists incompatible format"

            if mode == 'QA':
                assert len(gt_feature) == 1, f"gt {id} {condition} {name} exists one more answer"
                assert len(gen_feature) == 1, f"gen {id} {condition} {name} exists one more answer"
                # list to str
                gt_content = str(gt_feature[0]).lower()
                gen_content = str(gen_feature[0]).lower()
            elif mode == 'IE':
                gt_content = [str(x).lower() for x in gt_feature]
                gen_content = [str(x).lower() for x in gen_feature]             

            temp_data.append({
                "study_id": id,
                "condition": condition,
                "gt_feature": gt_content,
                "gen_feature": gen_content
            })
            appended = True
            
        # If there is no true positive between gt and gen / all skipped, still add a row
        if not appended:
            temp_data.append({
                "study_id": id,
                "condition": None,
                "gt_feature": None,
                "gen_feature": None
            })
            continue
    
    df_feature = pd.DataFrame(temp_data)
    return df_feature


def clamp_score(value: float) -> float:
    """Clamp a floating point value into the [0, 1] interval."""
    return max(0.0, min(1.0, value))


def interpret_llm_score(response: str) -> Optional[float]:
    """Attempt to coerce an LLM response into a numeric score between 0 and 1."""
    if not response:
        return None

    score_tag_match = re.search(r'</?SCORE>\s*"?(-?\d+(?:\.\d+)?)"?\s*</SCORE>', response, re.IGNORECASE)
    if score_tag_match:
        value = float(score_tag_match.group(1))
        return clamp_score(value)


def get_one_response(usr_prompt, sys_prompt, model_name="gpt-4o-mini"):
    """
    Get a response from Azure OpenAI API
    
    Args:
        usr_prompt: User prompt text
        sys_prompt: System prompt text
        model_name: Name of the model to use (must be defined in MODEL_CONFIGS)
        
    Returns:
        Response text from the model
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not found in MODEL_CONFIGS.")
    
    config = MODEL_CONFIGS[model_name]
    
    client = AzureOpenAI(
        api_key=config["api_key"],
        api_version=config["api_version"],
        azure_endpoint=config["endpoint"]
    )

    max_tokens = config.get("max_tokens")

    try:
        request_payload = {
            "model": config["deployment"],
            "messages": [
                {
                    "role": "system", 
                    "content": sys_prompt
                },
                {
                    "role": "user",
                    "content": usr_prompt
                }
            ]
        }
        if max_tokens is not None:
            request_payload["max_tokens"] = max_tokens

        apiresponse = client.chat.completions.with_raw_response.create(**request_payload)
    except:
        fallback_payload = {
            "model": config["deployment"],
            "messages": [
                {
                    "role": "user",
                    "content": sys_prompt + "\n\n" + usr_prompt
                }
            ]
        }
        if max_tokens is not None:
            fallback_payload["max_tokens"] = max_tokens

        apiresponse = client.chat.completions.with_raw_response.create(**fallback_payload)

    chat_completion = apiresponse.parse()
    response = chat_completion.choices[0].message.content
    
    return response


def compute_acc_mirco(gt: pd.Series, gen: pd.Series) -> float:
    '''
    Accuracy across all features
    '''
    gt = gt.dropna()
    gen = gen.dropna()
    
    gt_list = gt.tolist()
    gen_list = gen.tolist()
    correct = sum(1 for gt, gen in zip(gt_list, gen_list) if gt == gen)
    return round(correct / len(gt_list), 3) if gt_list else float("nan")

def compute_acc_macro(df: pd.DataFrame) -> float:
    '''
    Accuracy across all conditions
    '''
    df = df.dropna(subset=["condition"])
    
    conditions = df['condition'].unique()
    scores = []

    for condition in conditions:
        temp_df = df[df['condition'] == condition]
        gt_list = temp_df['gt_feature'].tolist()
        gen_list = temp_df['gen_feature'].tolist()
        score = (sum(1 for gt, gen in zip(gt_list, gen_list) if gt == gen)) / len(gt_list) if gt_list else float("nan")

        if not math.isnan(score):
            scores.append(score)

    return round(sum(scores) / len(scores), 3) if scores else float("nan")

def compute_acc_per_row(df: pd.DataFrame) -> pd.Series:
    def _acc(row) -> float:
        if pd.isna(row["condition"]):
            return 0.0
        return float(row["gt_feature"] == row["gen_feature"])

    return df.apply(_acc, axis=1)

def preprocess(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def bleu_score(gt, gen):
    '''
    BLEU-4 computation
    '''
    smoothie = SmoothingFunction().method4
    return sentence_bleu([gt], gen, smoothing_function=smoothie)

def rouge_l_score(gt, gen):
    '''
    ROUGE-L computation
    '''
    gt_text = " ".join(gt)
    gen_text = " ".join(gen)
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return rouge.score(gt_text, gen_text)['rougeL'].fmeasure


def compute_llm_metric(
    gt_series: pd.Series,
    gen_series: pd.Series,
    llm_config: dict,
    return_details: bool = False,
    context: Optional[pd.DataFrame] = None
) -> Tuple[float, Optional[pd.DataFrame]]:
    """Leverage an LLM to score generated features against ground truth.

    Returns an average score between 0 and 1 (NaN if none could be parsed) and
    optionally a DataFrame capturing per-sample responses.
    """
    total = len(gt_series)
    if total != len(gen_series):
        raise ValueError("LLM scoring requires gt_series and gen_series to have equal length")

    show_progress = llm_config.get('show_progress', False)
    index_iterable = range(total)
    if show_progress:
        index_iterable = tqdm(index_iterable, desc="LLM scoring", leave=False)

    numeric_scores = []
    detail_rows = [] if return_details else None

    for idx in index_iterable:
        gt_list = gt_series.iloc[idx]
        gen_list = gen_series.iloc[idx]

        gt_text = json.dumps(gt_list, ensure_ascii=False)
        gen_text = json.dumps(gen_list, ensure_ascii=False)

        usr_template = llm_config['usr_prompt']
        formatted_usr_prompt = usr_template.format(groundtruth=gt_text, candidate=gen_text)
        response = get_one_response(formatted_usr_prompt, llm_config['sys_prompt'], llm_config['model_name'])
        numeric = interpret_llm_score(response)

        if numeric is not None:
            numeric_scores.append(numeric)

        if return_details:
            context_data = {}
            if context is not None:
                if hasattr(context, 'iloc'):
                    context_row = context.iloc[idx]
                    if hasattr(context_row, 'to_dict'):
                        context_data = context_row.to_dict()
                elif isinstance(context, list):
                    context_data = context[idx]

            detail_rows.append({
                **context_data,
                'gt_feature': gt_list,
                'gen_feature': gen_list,
                'llm_response': response,
                'llm_score': numeric
            })

    mean_score = round(float(np.mean(numeric_scores)), 3) if numeric_scores else float('nan')

    if return_details:
        return mean_score, pd.DataFrame(detail_rows)

    return mean_score, None


def compute_similarity(
    gt_series,
    gen_series,
    metric: str = 'rouge',
    llm_config: Optional[dict] = None,
    return_details: bool = False,
    context: Optional[pd.DataFrame] = None
):
    '''
    Compute similarity metrics between ground truth and generated feature lists.
    
    Supports traditional lexical metrics (ROUGE-L, BLEU-4) as well as optional
    LLM-judged scoring when ``metric`` is set to ``'llm'``.
    '''
    if metric == 'llm':
        if llm_config is None:
            raise ValueError("LLM similarity scoring requires an llm_config")
        score, details = compute_llm_metric(
            gt_series=gt_series,
            gen_series=gen_series,
            llm_config=llm_config,
            return_details=return_details,
            context=context
        )
        return (score, details) if return_details else score

    all_scores = []
    for gt_list, gen_list in zip(gt_series, gen_series):
        scores = []
        
        # if no true positive between gt and gen
        if gt_list is None:
            all_scores.append(0.0)
            continue
            
        for gt in gt_list:
            gt_tokens = preprocess(gt)
            if not gt_tokens:
                scores.append(0.0)
                continue

            best_score = 0.0
            for gen in gen_list:
                gen_tokens = preprocess(gen)
                if not gen_tokens:
                    continue

                if metric == 'bleu':
                    score = bleu_score(gt_tokens, gen_tokens)
                else:  # default: rouge
                    score = rouge_l_score(gt_tokens, gen_tokens)

                best_score = max(best_score, score)
            scores.append(best_score)
        all_scores.append(np.mean(scores))

    return round(np.mean(all_scores), 3) if all_scores else float('nan'), all_scores

def cal_metrics(df_feature, name, mode, llm_config: Optional[dict] = None, skip_llm: bool = False):
    '''
    Calculate appropriate metrics based on feature mode (QA or IE)
    '''            
    if mode == 'QA':
        # 1. Acc. (micro)
        acc_micro = compute_acc_mirco(df_feature['gt_feature'], df_feature['gen_feature'])

        # 2. Acc. (macro)
        acc_macro = compute_acc_macro(df_feature)
        
        acc_per_report = compute_acc_per_row(df_feature)

        metric_dict = {
            'Feature': name,
            'Acc. (micro)': acc_micro,
            'Acc. (macro)': acc_macro       
        }
        
        df_feature['acc'] = acc_per_report
        
        df_metric_per_report = df_metric_per_report = (
            df_feature.groupby("study_id", sort=False)[["acc"]]
            .mean()
            .reset_index()
        )
        
        return metric_dict, df_metric_per_report, None

    if mode == 'IE':
        # 1. ROUGE-L
        rouge, per_row_rouges = compute_similarity(df_feature['gt_feature'], df_feature['gen_feature'], metric='rouge')

        # 2. BLEU-4
        bleu, per_row_bleus = compute_similarity(df_feature['gt_feature'], df_feature['gen_feature'], metric='bleu')

        metric_dict = {
            'Feature': name,
            'ROUGE-L': rouge,
            'BLEU-4': bleu
        }
        
        df_feature['rouge'] = per_row_rouges
        # df_feature['bleu'] = per_row_bleus
        
        df_metric_per_report = (
            df_feature.groupby("study_id", sort=False)[["rouge"]]
            .mean()
            .reset_index()
        )
        
        llm_details = None
        if llm_config and not skip_llm:
            llm_score, llm_details = compute_similarity(
                df_feature['gt_feature'],
                df_feature['gen_feature'],
                metric='llm',
                llm_config=llm_config,
                return_details=True,
                context=df_feature[['study_id', 'condition']]
            )
            metric_dict['LLM Score'] = llm_score

        return metric_dict, df_metric_per_report, llm_details

    raise ValueError(f"Unsupported mode '{mode}' for metric calculation")

def evaluate_qa_features(
    dict_gt,
    dict_gen,
    metric_pth,
    model_name: Optional[str] = None
):
    '''
    Evaluate QA-type features
    '''
    print("Evaluating QA features...")
    metric_data = []
    mode = 'QA'
    
    per_report_metric_dfs = []
    for name in FEATURE_CONFIG[mode]:
        print(f"Processing {name}...")
        # 1. transform df
        df_feature = convert_feature_df(dict_gt, dict_gen, name, mode)

        # 2. calculate metrics
        metric_dict, per_report_metric_df, _ = cal_metrics(df_feature, name, mode)

        # 3. output metric
        metric_data.append(metric_dict)
        
        # 3.5. per-report metric
        per_report_metric_df = per_report_metric_df\
            .rename(columns={"acc": name})\
            .set_index("study_id")
        per_report_metric_dfs.append(per_report_metric_df)

    metric_df = pd.DataFrame(metric_data)
    os.makedirs(metric_pth, exist_ok=True)
    output_filename = f'results_qa_avg_{model_name}.csv'

    output_file = os.path.join(metric_pth, output_filename)
    metric_df.to_csv(output_file, index=False)
    print(f"QA evaluation results saved to {output_file}")
    
    df_all_qa_metrics_per_report = pd.concat(per_report_metric_dfs, axis=1).reset_index()
    output_file_per_report = os.path.join(metric_pth, f"results_qa_per_report_{model_name}.csv")
    df_all_qa_metrics_per_report.to_csv(output_file_per_report, index=False)
    print(f"QA per-report evaluation results saved to {output_file_per_report}")
    return metric_df

def evaluate_ie_features(
    dict_gt,
    dict_gen,
    metric_pth,
    model_name: Optional[str] = None,
    llm_config: Optional[dict] = None,
    skip_llm: bool = False
):
    '''
    Evaluate IE-type features
    '''
    print("Evaluating IE features...")
    metric_data = []
    mode = 'IE'
    
    per_report_metric_dfs = []
    for name in FEATURE_CONFIG[mode]:
        print(f"Processing {name}...")
        # 1. transform df
        df_feature = convert_feature_df(dict_gt, dict_gen, name, mode)

        # 2. calculate metrics
        metric_dict, per_report_metric_df, llm_details = cal_metrics(
            df_feature,
            name,
            mode,
            llm_config=llm_config,
            skip_llm=skip_llm
        )

        # 3. output metric
        metric_data.append(metric_dict)
        
        # 3.5. per-report metric: Only use ROUGE
        per_report_metric_df = per_report_metric_df\
            .rename(columns={"rouge": name})\
            .set_index("study_id")
        per_report_metric_dfs.append(per_report_metric_df)

        # 4. detailed llm eval output
        if llm_details is not None and not llm_details.empty:
            os.makedirs(metric_pth, exist_ok=True)
            slug = name.lower().replace(' ', '_')
            llm_filename = f'ie_llm_evaluation_{slug}_{model_name}.csv'
            llm_output_file = os.path.join(metric_pth, llm_filename)
            llm_details.to_csv(llm_output_file, index=False)
            print(f"LLM evaluation details saved to {llm_output_file}")

    metric_df = pd.DataFrame(metric_data)
    os.makedirs(metric_pth, exist_ok=True)
    output_filename = f'results_ie_avg_{model_name}.csv'
    output_file = os.path.join(metric_pth, output_filename)
    metric_df.to_csv(output_file, index=False)
    print(f"IE evaluation results saved to {output_file}")
    
    df_all_ie_metrics_per_report = pd.concat(per_report_metric_dfs, axis=1).reset_index()
    output_file_per_report = os.path.join(metric_pth, f"results_ie_per_report_{model_name}.csv")
    df_all_ie_metrics_per_report.to_csv(output_file_per_report, index=False)
    print(f"IE per-report evaluation results saved to {output_file_per_report}")
    
    return metric_df

def main():
    parser = argparse.ArgumentParser(description='Evaluate feature extraction metrics')
    parser.add_argument('--gen_path', type=str, required=True, 
                        help='Path to generated feature output JSON file')
    parser.add_argument('--gt_path', type=str, required=True, 
                        help='Path to ground truth feature JSON file')
    parser.add_argument('--metric_path', type=str, default='./metrics',
                        help='Directory to save metric results')
    parser.add_argument('--enable_llm_metric', action='store_true',
                        help='Include LLM-based scoring as an additional IE metric')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Identifier for the model that produced the generated features')

    # New command line arguments for model comparison
    parser.add_argument('--scoring_llm', type=str, default='gpt-4o-mini',
                        help='Model name to use for scoring')                        
    
    args = parser.parse_args()

    llm_config = None

    if args.enable_llm_metric:
        if args.scoring_llm not in MODEL_CONFIGS:
            raise ValueError(f"Scoring model '{args.scoring_llm}' not found in MODEL_CONFIGS.")

        usr_prompt_template = LLMMetricPrompts.USER_PROMPT_TEMPLATE
        sys_prompt_template = LLMMetricPrompts.SYSTEM_PROMPT


        llm_config = {
            "usr_prompt": usr_prompt_template,
            "sys_prompt": sys_prompt_template,
            "model_name": args.scoring_llm,
            "show_progress": True,
        }

    # Original evaluation functionality
    # Load input files
    print(f"Loading ground truth from {args.gt_path}...")
    dict_gt = load_json_file(args.gt_path, "ground truth data")

    print(f"Loading generated features from {args.gen_path}...")
    dict_gen = load_json_file(args.gen_path, "generated features")
    
    assert len(dict_gt) == len(dict_gen), "Length of ground truth and generated features mismatch!"
    
    # Evaluate features
    qa_results = evaluate_qa_features(
        dict_gt,
        dict_gen,
        args.metric_path,
        model_name=args.model_name
    )
    ie_results = evaluate_ie_features(
        dict_gt,
        dict_gen,
        args.metric_path,
        model_name=args.model_name,
        llm_config=llm_config
    )

    # Print summary results
    print("\nQA Evaluation Summary:")
    print(qa_results)
    
    print("\nIE Evaluation Summary:")
    print(ie_results)

if __name__ == "__main__":
    main()
