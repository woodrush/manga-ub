import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import model_names, get_latex_table, circular_correctness_multichoice


basedir = "bench_results/recognition_background/"
table_outpath = "eval_results/table/recognition_background.tex"


d_n_expected_rows = {
    "Location": 2,
    "Weather": 2,
    "Time_of_day": 2,
    "Weather_difficult": 3,
}

def structure_check(df):
    task, label, _ = df.name
    n_expected_rows = d_n_expected_rows[task]
    assert df.shape[0] == n_expected_rows, df
    assert df.expected.nunique() == len(df.expected), df


if __name__ == "__main__":
    l_stats = []
    for filename in tqdm(os.listdir(basedir)):
        model_name = filename.split("-")[0]
        if model_name not in model_names:
            print(f"Skipping {model_name}")
            continue

        df_responses = pd.read_csv(f"{basedir}/{filename}")
        df_responses = df_responses.rename({
            "circular_index": "circular_id"
        }, axis=1)
        df_responses["model_name"] = model_name

        df_responses.groupby(["task", "label", "circular_id"]).apply(structure_check, include_groups=False)

        df_circ_aggregated = df_responses.groupby(["model_name", "task", "label", "circular_id"]).apply(
            circular_correctness_multichoice, include_groups=False
        )
        df_label_aggregated = df_circ_aggregated.groupby(["model_name", "task", "label"]).apply(np.mean, include_groups=False)
        df_task_aggregated = df_label_aggregated.groupby(["model_name", "task"]).apply(np.mean, include_groups=False)
        l_stats.append(df_task_aggregated)

    df_stats = pd.concat(l_stats).unstack("model_name")

    os.makedirs("eval_results/table", exist_ok=True)
    table = get_latex_table(df_stats)
    table = f"% Background recognition\n{table}\n\n"
    print(table)
    with open(table_outpath, "wt") as f:
        f.write(table)
