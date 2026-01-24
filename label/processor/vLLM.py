import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import re
from vllm import LLM, SamplingParams
import json
from configs.models import MODEL_CONFIGS
from configs.prompts import SYS_PROMPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, dest='model_name',
                        default=None,
                        help="Model name to select from MODEL_CONFIGS.")
    parser.add_argument("--reports", type=str, 
                        dest="input_csv", default=None, 
                        help="Path to input CSV file containing reports.")
    parser.add_argument("--output", type=str, 
                        dest='output_dir', default=None, 
                        help="Directory path to output results.")
    args = parser.parse_known_args()

    return args


class vLLMProcessor:
    def __init__(self, model_name, input_csv, output_dir):
        self.model = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.in_csv = input_csv
        self.out_dir = output_dir
        self.sampling_params, self.tokenizer, self.llm = self.prepare_llm()

    def prepare_llm(self):
        '''
        Initialize vLLM model
        '''
        sampling_params = SamplingParams(temperature=self.config['temperature'], max_tokens=self.config['max_tokens'])
        llm = LLM(self.config["model_path"], tensor_parallel_size=self.config["tensor_parallel_size"], max_model_len=4096)
        tokenizer = llm.get_tokenizer()

        return sampling_params, tokenizer, llm

    def run_label_extraction(self, df_gt_repo):
        '''
        Extract labels using zero-shot prompting
        '''
        all_prompt = []
        prompt_s = SYS_PROMPT
        
        for _, row in df_gt_repo.iterrows():
            report = row['report']

            all_prompt.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": prompt_s},
                        {"role": "user", "content": report},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        
        ls_all_outputs = self.llm.generate(all_prompt, self.sampling_params)
        return ls_all_outputs

    def run(self):
        '''
        Main execution method
        '''
        # Step 1: Load Files
        print("Loading input data...")
        df_gt_repo = pd.read_csv(self.in_csv)
        df_gt_repo['study_id'] = df_gt_repo['study_id'].apply(lambda x: str(x))
        df_gt_repo = df_gt_repo.sort_values(by='study_id').reset_index(drop=True)
        task1_results = {}

        # Step 2: Label Extraction
        print("Running label extraction...")
        ls_all_outputs = self.run_label_extraction(df_gt_repo)

        assert len(ls_all_outputs) == len(df_gt_repo), "Mismatch between outputs and input data length."
        print('Finished label extraction.')
        print('Processing output...')

        # Step 3: Extract results
        for i, row in tqdm(df_gt_repo.iterrows(), total=len(df_gt_repo)):
            id = row['study_id']
            generated_text = ls_all_outputs[i].outputs[0].text
            print(generated_text)
            task1_match = re.search(r'<TASK1>(.*?)</TASK1>', generated_text, re.DOTALL)
            
            if task1_match:
                task1_content = task1_match.group(1).strip()
                task1_results[id] = json.loads(task1_content)
            else:
                print(f"Warning: No correct label match for ID {id}")
                task1_results[id] = generated_text
        
        # Step 4: Save files
        print("Saving results...")
        output_tmp_dir = os.path.join(self.out_dir, 'tmp')
        os.makedirs(output_tmp_dir, exist_ok=True)
        output_file = os.path.join(output_tmp_dir, f'output_labels_{self.model}.json')
        with open(output_file, 'w', encoding='utf-8') as task1_file:
            json.dump(task1_results, task1_file, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}")


if __name__ == '__main__':
    args, _ = parse_args()

    if args.model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{args.model_name}' not found in MODEL_CONFIGS.")

    if not args.input_csv:
        raise ValueError("Input CSV file must be specified with --input_csv")
    
    if not args.output_dir:
        raise ValueError("Output directory must be specified with --o")

    processor = vLLMProcessor(
        model_name=args.model_name,
        input_csv=args.input_csv,
        output_dir=args.output_dir
    )
    processor.run()
