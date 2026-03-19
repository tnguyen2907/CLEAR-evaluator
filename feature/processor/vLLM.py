import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import re
from vllm import LLM, SamplingParams
import json
from ..configs.prompts import PromptDict
from ..configs.models import MODEL_CONFIGS



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
        llm = LLM(
            self.config["model_path"],
            tensor_parallel_size=self.config["tensor_parallel_size"],
            data_parallel_size=self.config["data_parallel_size"],
            max_model_len=4096,
            gpu_memory_utilization=self.config["gpu_memory_utilization"],
            distributed_executor_backend="external_launcher",
        )
        tokenizer = llm.get_tokenizer()

        return sampling_params, tokenizer, llm


    def get_positive_conditions(self, labels):
        return labels[labels == 1].index.tolist() # positive: 1 # positive condition list for each study

    def run(self):
        dp_rank = self.llm.llm_engine.vllm_config.parallel_config.data_parallel_rank
        dp_size = self.llm.llm_engine.vllm_config.parallel_config.data_parallel_size

        # Step 1: Prepare Files
        df_repo = pd.read_csv(self.in_reports, dtype={'study_id':  str})\
            .sort_values(by='study_id')\
            .reset_index(drop=True) # str | str
        df_labels = pd.read_csv(self.in_labels, dtype={'study_id':  str})\
            .sort_values(by='study_id')\
            .reset_index(drop=True) # str | int | ... | int
        ls_id = df_repo['study_id'].unique()

        print(f"[DP rank {dp_rank}] input labels file: {self.in_labels}")

        # Step 2: Collect all prompts for batched inference
        all_prompt_dict = PromptDict.get_all_prompt()
        all_prompts = []       # flat list of formatted prompts
        prompt_keys = []       # (study_id, condition, feature) for each prompt
        output_dict = {}

        for study_id in ls_id:
            # Preserve empty studies so downstream consumers keep a 1:1 ID mapping.
            output_dict.setdefault(study_id, {})
            report = df_repo[df_repo['study_id'] == study_id]['report'].iloc[0]
            label_row = df_labels[df_labels["study_id"] == study_id].iloc[0].drop("study_id")
            ls_conditions = self.get_positive_conditions(label_row)

            for condition in ls_conditions:
                prompt_dict = all_prompt_dict[condition]
                for feature, prompt in prompt_dict.items():
                    full_prompt = self.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": report},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    all_prompts.append(full_prompt)
                    prompt_keys.append((study_id, condition, feature))

        # Step 3: Shard prompts by DP rank
        cur_indices = list(range(dp_rank, len(all_prompts), dp_size))
        cur_prompts = [all_prompts[i] for i in cur_indices]
        cur_keys = [prompt_keys[i] for i in cur_indices]
        print(f"[DP rank {dp_rank}] Processing {len(cur_prompts)}/{len(all_prompts)} prompts")

        # Step 4: Batched inference — single llm.generate() call
        cur_outputs = self.llm.generate(cur_prompts, self.sampling_params)

        # Step 5: Map results back to nested dict structure
        for (study_id, condition, feature), output in zip(cur_keys, cur_outputs):
            generated_text = output.outputs[0].text
            match_feature = re.search(r'(\[.*?\])', generated_text, re.DOTALL)
            result = match_feature.group(1) if match_feature else "[\"NaN\"]"

            if study_id not in output_dict:
                output_dict[study_id] = {}
            if condition not in output_dict[study_id]:
                output_dict[study_id][condition] = {}
            output_dict[study_id][condition][feature] = result

        # Step 6: Save rank-specific results
        os.makedirs(os.path.join(self.out_dir, "tmp"), exist_ok=True)
        rank_file = os.path.join(self.out_dir, f"tmp/output_feature_{self.model}_rank{dp_rank}.json")
        with open(rank_file, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=4)

        # Step 7: Barrier — wait for all ranks to finish writing
        import torch.distributed as dist
        dist.barrier()

        # Step 8: Rank 0 merges all rank files into final output (sorted by study_id)
        if dp_rank == 0:
            merged = {}
            for rank in range(dp_size):
                rfile = os.path.join(self.out_dir, f"tmp/output_feature_{self.model}_rank{rank}.json")
                with open(rfile, encoding="utf-8") as f:
                    rank_data = json.load(f)
                # Deep merge: study_id -> condition -> feature
                for sid, conditions in rank_data.items():
                    if sid not in merged:
                        merged[sid] = {}
                    for cond, features in conditions.items():
                        if cond not in merged[sid]:
                            merged[sid][cond] = {}
                        merged[sid][cond].update(features)
                os.remove(rfile)
            merged = dict(sorted(merged.items()))
            output_path = os.path.join(self.out_dir, f"tmp/output_feature_{self.model}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=4)
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

    # Explicit cleanup: vLLM workers don't reliably release CUDA IPC handles at shutdown
    import contextlib, gc, torch
    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    del Processor
    gc.collect()
    torch.cuda.empty_cache()
