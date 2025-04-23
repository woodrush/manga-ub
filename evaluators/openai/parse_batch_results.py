import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import json
import time


d_csv_paths = {
    "character_count":                       "tasks/character_count.csv",
    "emotion_benchmark_split":               "tasks/emotion_benchmark_split.csv",
    "onomatopoeia_scene":                    "tasks/onomatopoeia_scene.csv",
    "panel_localization":                    "tasks/panel_localization.csv",
    "recognition_background":                "tasks/recognition_background.csv",
    "next_panel_inference":                  "tasks/next_panel_inference/test.csv",
}

batch_output_basedir = "batch_tasks_output/openai"
out_basedir = "bench_results"


def parse_response(response):
    prompt_id = response["custom_id"]
    response = response["response"]
    r = response["body"]["choices"][0]["message"]["content"]
    prompt_tokens = response["body"]["usage"]["prompt_tokens"]
    completion_tokens = response["body"]["usage"]["completion_tokens"]
    return {
        "prompt_id": prompt_id,
        "response_raw": r,
        "response": r,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


if __name__ == "__main__":
    for taskname, csv_path in d_csv_paths.items():
        # Read the tasks results
        taskdir = f"{batch_output_basedir}/{taskname}"
        l_responses = []
        print(taskname)
        for filename in tqdm(os.listdir(taskdir)):
            filepath = f"{taskdir}/{filename}"
            with open(filepath, "rt") as f:
                s = f.read()
                l = [parse_response(json.loads(r)) for r in s.split("\n") if len(r) > 0]
            l_responses += l
        df_responses = pd.DataFrame(l_responses)

        # Read the original CSV input
        df_input = pd.read_csv(csv_path)

        assert set(df_responses.prompt_id.unique()) == set(df_input.prompt_id.unique())
        assert df_responses.shape[0] == df_input.shape[0]

        df_output = pd.merge(df_input, df_responses, on="prompt_id", how="outer")
        assert df_output.shape[0] == df_input.shape[0]

        outdir = f"{out_basedir}/{taskname}"
        os.makedirs(outdir, exist_ok=True)
        outpath = f"{outdir}/gpt4o-batch.csv"
        df_output.to_csv(outpath)
