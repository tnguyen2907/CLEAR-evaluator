import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import re
# from openai import AzureOpenAI
from vllm import LLM, SamplingParams
import json
from configs.prompts import PromptDict
from configs.models import MODEL_CONFIGS



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, dest='model_name', 
                        default=None, 
                        help="Model name to select from MODEL_CONFIGS.") ### TO-DO: set gpt-4o-mini as default
    parser.add_argument("--reports", type=str, 
                        dest="input_reports", default=None, 
                        help="Directory path to input text reports.") ### TO-DO: set gt report as default
    parser.add_argument("--labels", type=str, 
                        dest="input_labels", default=None, 
                        help="Directory path to input condition labels.") ### TO-DO: set gt labels as default
    parser.add_argument("--output", type=str, 
                        dest='output_dir', default=None, 
                        help="Directory path to output csv(s).")
    args = parser.parse_known_args()

    return args


class vLLMProcessor:
    def __init__(self, model_name, input_reports, input_labels, output_dir):
        self.model = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.sampling_params, self.tokenizer, self.llm = self.prepare_llm()

        self.in_reports = input_reports
        self.in_labels = input_labels
        self.out_dir = output_dir


    def prepare_llm(self):
        '''
        Unique for vllm
        '''
        sampling_params = SamplingParams(temperature=self.config["temperature"], max_tokens=self.config["max_tokens"])
        llm = LLM(self.config["model_path"], tensor_parallel_size=self.config["tensor_parallel_size"], max_model_len=4096)
        tokenizer = llm.get_tokenizer()

        return sampling_params, tokenizer, llm


    def get_one_response(self, report, prompt):
        '''
        Unique for vllm
        '''
        full_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": report},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        
        output = self.llm.generate([full_prompt], self.sampling_params)
        response = output[0].outputs[0].text

        return response

    def get_positive_conditions(self, labels):
        return labels[labels == 1].index.tolist() # positive: 1 # positive condition list for each study


    def run_feature_extraction(self, report, labels):
        # Step 1: Load Prompt Dict
        all_prompt_dict = PromptDict.get_all_prompt()
        ls_conditions = self.get_positive_conditions(labels)
        all_feature_dict = {} # layer 1: condition; layer 2: feature
        

        # Step 2: Run Each Prompt Request
        for condition in ls_conditions:
            temp_feature_dict = {}
            prompt_dict = all_prompt_dict[condition]

            for feature, prompt in prompt_dict.items():
                generated_text = self.get_one_response(report, prompt)
                # print(generated_text)
                match_feature = re.search(r'(\[.*?\])', generated_text, re.DOTALL)
                temp_feature_dict[feature] = match_feature.group(1) if match_feature else "[\"NaN\"]" ### "Format Error" ## TO-DO: return one error list

            all_feature_dict[condition] = temp_feature_dict
        
        return all_feature_dict


    def run(self):
        # Step 1: Prepare Files
        df_repo = pd.read_csv(self.in_reports, dtype={'study_id':  str})\
            .sort_values(by='study_id')\
            .reset_index(drop=True) # str | str
        df_labels = pd.read_csv(self.in_labels, dtype={'study_id':  str})\
            .sort_values(by='study_id')\
            .reset_index(drop=True) # str | int | ... | int
        ls_id = df_repo['study_id'].unique()
        output_dict = {}
        
        print("input labels file:", self.in_labels)

        # Step 2: Iterate and extract
        for id in tqdm(ls_id):
            report = df_repo[df_repo['study_id'] == id]['report'].iloc[0]
            label_row = df_labels[df_labels["study_id"] == id].iloc[0].drop("study_id")
            output_dict[id] = self.run_feature_extraction(report, label_row)

        # Step 3: Save json
        output_path = os.path.join(self.out_dir, f"tmp/output_feature_{self.model}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=4)
        print(f"Saved output to {output_path}")



if __name__ == '__main__':
    args, _ = parse_args()

    if args.model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{args.model_name}' not found in MODEL_CONFIGS.")

    Processor = vLLMProcessor(
        model_name = args.model_name,
        input_reports=args.input_reports,
        input_labels=args.input_labels,
        output_dir=args.output_dir
    )
    Processor.run()