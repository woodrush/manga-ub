import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import time


if __name__ == "__main__":
    basedir = "batch_tasks_output/openai/failed"

    l_failed = []
    for filename_failed in os.listdir(basedir):
        print(filename_failed)
        filepath = f"{basedir}/{filename_failed}"
        l_j = []
        with open(filepath, "rt") as f:
            for line in f.readlines():
                j = json.loads(line)
                l_j.append(j)
        l = [j["custom_id"] for j in l_j]
        taskname, input_json_filename = filename_failed.split("__")
        l_failed.append({
            "taskname": taskname,
            "input_json_filename": input_json_filename,
            "l_prompt_id": l
        })

    input_basedir = "batch_tasks/openai/input_split"

    l_round2_batches = []
    for d in l_failed:
        taskname = d["taskname"]
        input_json_filename = d["input_json_filename"]
        l_prompt_id = d["l_prompt_id"]
        l_lines = []

        input_json_filepath = f"{input_basedir}/{taskname}/{input_json_filename}"
        with open(input_json_filepath, "rt") as f:
            for line in f.readlines():
                for prompt_id in l_prompt_id:
                    if prompt_id in line:
                        l_lines.append(line)
                if len(l_lines) == len(l_prompt_id):
                    break

        l_round2_batches.append({
            "taskname": taskname,
            "input_json_filename": input_json_filename,
            "l_prompts": l_lines,
        })

    basedir = "batch_tasks/openai/input_split_failed_round2"
    os.makedirs(basedir, exist_ok=True)
    for r2batch in l_round2_batches:
        taskname = r2batch["taskname"]
        input_json_filename = r2batch["input_json_filename"].split(".jsonl")[0]
        l_prompts = r2batch["l_prompts"]
        json_filename = f"{input_json_filename}_failed_round2.jsonl"
        d = f"{basedir}/{taskname}"
        os.makedirs(d, exist_ok=True)
        outpath = f"{d}/{json_filename}"

        s = "\n".join(l_prompts)
        with open(outpath, "wt") as f:
            f.write(s)
