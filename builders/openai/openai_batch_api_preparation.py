import pandas as pd
import numpy as np
import os

import sys
sys.path.append(".")

from tqdm import tqdm

from models.gpt4o import GPT4o
from multiprocessing import Pool

from configs import d_csv_paths

build_batch_out_dir = "batch_tasks/openai/input"

N_PROCESSES = 16

def process_row(item):
    _, row = item
    jsonl_line = GPT4o.generate_batch_api_message(row.prompt, row.impath, custom_id=row.prompt_id)
    return jsonl_line

if __name__ == "__main__":
    os.makedirs(build_batch_out_dir, exist_ok=True)

    with Pool(processes=N_PROCESSES) as pool:
        for task_name, csv_path in d_csv_paths.items():
            out_build_jsonl_path = f"{build_batch_out_dir}/{task_name}.jsonl"
            print(f"Writing {out_build_jsonl_path} ...")
            with open(out_build_jsonl_path, "wt") as f:
                df_prompts = pd.read_csv(csv_path)
                process_prompts_iter = pool.imap(process_row, df_prompts.iterrows())

                for result in tqdm(process_prompts_iter, total=df_prompts.shape[0]):
                    f.write(result + "\n")
