import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import re
from openai import AzureOpenAI
import json
from ..configs.models import MODEL_CONFIGS
from ..configs.prompts import SYS_PROMPT


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


class AzureProcessor:
    def __init__(self, model_name, input_csv, output_dir):
        self.model = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.in_csv = input_csv
        self.out_dir = output_dir

    def get_one_response(self, report, system_prompt):
        '''
        Get response from Azure OpenAI API
        '''
        client = AzureOpenAI(
            api_key=self.config["api_key"],
            api_version=self.config["api_version"],
            azure_endpoint=self.config["endpoint"]
        )

        max_tokens = self.config.get("max_tokens")
        request_payload = {
            "model": self.config["deployment"],
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": report,
                },
            ],
        }

        if max_tokens is not None:
            request_payload["max_tokens"] = max_tokens

        try:
            chat_completion = client.chat.completions.create(**request_payload)
        except Exception as first_exc:
            fallback_payload = request_payload.copy()
            fallback_payload["messages"] = [
                {
                    "role": "user",
                    "content": f"{system_prompt}\n\n{report}",
                }
            ]

            try:
                chat_completion = client.chat.completions.create(**fallback_payload)
            except Exception as second_exc:
                raise RuntimeError(
                    "Azure chat completion failed for both standard and fallback prompts"
                ) from second_exc

        response = chat_completion.choices[0].message.content
        
        return response

    def run_label_extraction(self, df_gt_repo):
        '''
        Extract labels using zero-shot prompting
        '''
        results = []
        prompt_s = SYS_PROMPT

        for _, row in df_gt_repo.iterrows():
            report = row['report']

            response = self.get_one_response(report, prompt_s)
            results.append(response)
        
        return results

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
        responses = self.run_label_extraction(df_gt_repo)

        assert len(responses) == len(df_gt_repo), "Mismatch in number of responses and input reports."
        print('Finished label extraction.')
        print('Processing output...')

        # Step 3: Extract results
        for i, row in tqdm(df_gt_repo.iterrows(), total=len(df_gt_repo)):
            id = row['study_id']
            generated_text = responses[i]
            task1_match = re.search(r'<TASK1>(.*?)</TASK1>', generated_text, re.DOTALL)
            
            if task1_match:
                task1_content = task1_match.group(1).strip()
                # Azure responses often wrap the JSON in Markdown fences; strip them before parsing
                if task1_content.startswith("```"):
                    task1_content = re.sub(r"^```(?:[a-zA-Z0-9_+-]+)?\s*", "", task1_content)
                    task1_content = re.sub(r"\s*```$", "", task1_content)

                try:
                    task1_results[id] = json.loads(task1_content)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON for ID {id}, storing raw content")
                    task1_results[id] = task1_content
            else:
                print(f"Warning: No TASK1 match for ID {id}")
                task1_results[id] = generated_text
        
        # Step 4: Save files
        print("Saving results...")
        os.makedirs(os.path.join(self.out_dir, 'tmp'), exist_ok=True)
        output_file = os.path.join(self.out_dir, 'tmp', f'output_labels_{self.model}.json')
        with open(output_file, 'w', encoding='utf-8') as task1_file:
            json.dump(task1_results, task1_file, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_file}")


if __name__ == '__main__':
    args, _ = parse_args()
    print(args.model_name)

    if args.model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{args.model_name}' not found in MODEL_CONFIGS.")

    if not args.input_csv:
        raise ValueError("Input CSV file must be specified with --input_csv")
    
    if not args.output_dir:
        raise ValueError("Output directory must be specified with --o")

    processor = AzureProcessor(
        model_name=args.model_name,
        input_csv=args.input_csv,
        output_dir=args.output_dir
    )
    processor.run()
