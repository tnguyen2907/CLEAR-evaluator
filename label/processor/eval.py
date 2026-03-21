import ast
import glob
import json
import os
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CXR_LABELS_1 = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
    'Fracture', 'Support Devices',
]

# Computes negative F1 and negative F1-5 for the labels:
# Edema, Consolidation, Pneumonia, Pneumothorax, Pleural Effusion.
# Also returns a list of Negative F1's for each label
def negative_f1(gt, pred):
    labels = range(13)
    labels_five = list(map(lambda x: CXR_LABELS_1.index(x), 
                           ["Edema", "Consolidation", "Pneumonia", 
                            "Pneumothorax", "Pleural Effusion"]))
    f1_scores = []

    for i in labels:
        score = f1_score(gt[:, i], pred[:, i], zero_division=0)
        f1_scores.append(score)
    f1_scores = np.array(f1_scores)

    neg_f1 = f1_scores.mean()
    neg_f1_five = f1_scores[labels_five].mean()
    micro_f1 = f1_score(gt.flatten(), pred.flatten(), zero_division=0, average='micro')
    return neg_f1, neg_f1_five, micro_f1, f1_scores

# Computes positive F1 and positive F1-5 for all labels except No Finding
# When `use_five` is True, we only calculate F1 with the labels:
# Atelectasis, Consolidation, Edema, Pleural Effusion, Cardiomegaly
def positive_f1(gt, pred):
    labels = range(13)
    labels_five = list(map(lambda x: CXR_LABELS_1.index(x), 
                           ["Cardiomegaly", "Edema", "Consolidation", 
                            "Atelectasis", "Pleural Effusion"]))
    f1_scores = []

    for i in labels:
        score = f1_score(gt[:, i], pred[:, i], zero_division=0)
        f1_scores.append(score)
    f1_scores = np.array(f1_scores)

    pos_f1 = f1_scores.mean()
    pos_f1_five = f1_scores[labels_five].mean()
    micro_f1 = f1_score(gt.flatten(), pred.flatten(), zero_division=0, average='micro')
    return pos_f1, pos_f1_five, micro_f1, f1_scores

# Computes the positive and negative F1 (excluding No Finding)
def compute_f1(df_gt, df_pred):
    y_gt = np.array(df_gt[CXR_LABELS_1]) 

    # Note on labels:
    # -1: unclear; 0: negative; 1: positive
    y_gt_neg = (y_gt == 0).astype(int)
    y_gt_pos = (y_gt == 1).astype(int)

    y_pred = np.array(df_pred[CXR_LABELS_1]) 

    y_pred_neg = (y_pred == 0).astype(int)
    y_pred_pos = (y_pred == 1).astype(int)

    pos_f1, pos_f1_five, pos_micro_f1, label_pos_f1 = positive_f1(y_gt_pos, y_pred_pos)
    neg_f1, neg_f1_five, neg_micro_f1, label_neg_f1 = negative_f1(y_gt_neg, y_pred_neg)
    return pos_f1, pos_f1_five, pos_micro_f1, neg_f1, neg_f1_five, neg_micro_f1, label_pos_f1, label_neg_f1

# Computes per-report positive F1
def compute_per_report_pos_f1(df_gt, df_pred):
    y_gt = np.array(df_gt[CXR_LABELS_1]) 
    y_gt_pos = (y_gt == 1).astype(int)

    y_pred = np.array(df_pred[CXR_LABELS_1])
    y_pred_pos = (y_pred == 1).astype(int)
    
    scores = []
    for i in range(y_gt_pos.shape[0]):
        s = f1_score(y_gt_pos[i], y_pred_pos[i], zero_division=0)
        scores.append(s)
        
    df_out = pd.DataFrame({
        "study_id": df_gt["study_id"].astype(str).values,
        "pos_f1_per_report": np.array(scores, dtype=float),
    })
    
    return df_out

def check_format_match(df_gen_labels): # check whether all convert to labels
    allowed_values = {0, 1, -1}
    temp_df = df_gen_labels[CXR_LABELS_1]
    invalid_condition = ~temp_df.isin(allowed_values)
    invalid_values = temp_df[invalid_condition]

    for index, row in invalid_values.iterrows():
        for col in row.index:
            if pd.notna(row[col]):
                print(f"Invalid value {row[col]} found at row {index}, column {col}")

def check_size_match(df_gt_labels, df_gen_labels): # one prerequiste: 'study_id' of df_gen_labels must be a subset of that in df_gt_labels
    set_gen = set(df_gen_labels['study_id'])
    set_gt = set(df_gt_labels['study_id'])
    if len(set_gen) < len(set_gt):
        set_missing  = set_gt.difference(set_gen)
        print('Missing generated samples', set_missing)
        df_gt_labels = df_gt_labels[df_gt_labels['study_id'].isin(list(set_gen))]
    elif len(set_gen) > len(set_gt):
        print("Alert: check why generated samples fail to be the subset of gt samples")
        df_gen_labels = df_gen_labels[df_gen_labels['study_id'].isin(list(set_gt))]
    
    return df_gt_labels, df_gen_labels         

def replace_values(value):
    replacements = {
        'positive': 1,
        'negative': 0,
        'unclear': -1,
        'n/a': -1,
        'nan': -1  # assuming you want to handle the string 'nan', not NaN values
    }
    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        try:
            value = ast.literal_eval(value)[0]
        except:
            pass  # In case of any error, keep the value unchanged

    return replacements.get(value, value)  # Return the replacement if it exists, otherwise the original value


if __name__ == '__main__':
    # Step 1: Prepare the input
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, help="Path to ground truth CSV file")
    parser.add_argument('--gen_dir', type=str, help="Directory containing generated result files")
    parser.add_argument('--model_name', type=str, default=None, help="Model name whose labels should be evaluated")
    args = parser.parse_args()
    gt_file = args.gt_dir
    gen_dir = args.gen_dir 
    label_dir = os.path.join(gen_dir, 'tmp')

    # Step 2: Load the data
    requested_filename = f'output_labels_{args.model_name}.json'
    label_path = os.path.join(label_dir, requested_filename)

    print(f"Loading generated results from: {label_path}")
    with open(label_path) as f:
        dict_all = json.load(f)
    
    rows = []
    ls_err_id = []
    dict_err = {}

    df_gt_labels = pd.read_csv(gt_file).sort_values(by='study_id')

    # Step 3: JSON to DataFrame (Generated data processing)
    for study_id, conditions in dict_all.items():
        row = {'study_id': study_id}
        if isinstance(conditions, str):
            try:
                conditions = json.loads(conditions) ### resolve format error 1: ought to be dict
            except:
                print(f"Warning: JSON format error")
                print(study_id)
                dict_err[study_id] = conditions ### save to solve other format errors
                ls_err_id.append(study_id)
                conditions = {c: -1 for c in CXR_LABELS_1}  # default to unclear (-1) so the study is not dropped
        
        for condition in CXR_LABELS_1:
            if condition in conditions: ### missing conditions
                row[condition] = conditions[condition]
            else:
                raise NotImplementedError(f"Condition '{condition}' not found in the generated data for study_id {study_id}")
        
        rows.append(row)
    
    df_gen = pd.DataFrame(rows)
    df_gen_labels = df_gen[['study_id']+CXR_LABELS_1]
    for col in CXR_LABELS_1:
        df_gen_labels[col] = df_gen_labels[col].apply(lambda x: str(x).lower())
    df_gen_labels = df_gen_labels.applymap(replace_values).sort_values(by='study_id').reset_index(drop=True)
    
    check_format_match(df_gen_labels)
    df_gt_labels, df_gen_labels = check_size_match(df_gt_labels, df_gen_labels)  

    # Step 4: Compute F1 scores and output metrics
    pos_f1, pos_f1_five, pos_micro_f1, neg_f1, neg_f1_five, neg_micro_f1, label_pos_f1, label_neg_f1 = compute_f1(df_gt_labels, df_gen_labels)

    metrics = {}
    metrics['pos f1'] = pos_f1
    metrics['pos f1_5'] = pos_f1_five
    metrics['pos micro f1'] = pos_micro_f1
    metrics['neg f1'] = neg_f1
    metrics['neg f1_5'] = neg_f1_five
    metrics['neg micro f1'] = neg_micro_f1
    
    for col in CXR_LABELS_1:
        metrics[col + ' (pos)'] = label_pos_f1[CXR_LABELS_1.index(col)]
    
    for col in CXR_LABELS_1:
        metrics[col + ' (neg)'] = label_neg_f1[CXR_LABELS_1.index(col)]
    
    df_metrics1 = pd.DataFrame(metrics, index=['model'])    
    
    # List all df_metrics in the terminal in a good format
    for col in df_metrics1.columns:
        print(f"{col}: {df_metrics1[col][0].round(4)}")
    
    # Save the DataFrame to a CSV file
    metrics_path = os.path.join(gen_dir, f'label_metrics_{args.model_name}.csv')
    df_metrics1.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    
    # Per-report metric
    df_per_report = compute_per_report_pos_f1(df_gt_labels, df_gen_labels)
    per_report_path = os.path.join(gen_dir, f'label_metrics_per_report_{args.model_name}.csv')
    
    df_per_report.to_csv(per_report_path, index=False)
    print(f"Per-report positive F1 saved to {per_report_path}")
    

    if len(dict_err) > 0:
        error_file = gen_dir + 'format_errors.json'
        print(f"Warning: {len(dict_err)} format errors found. Saving to {error_file}")
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(dict_err, f, ensure_ascii=False, indent=4)
