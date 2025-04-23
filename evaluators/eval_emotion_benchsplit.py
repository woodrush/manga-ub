import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import model_names, get_latex_table, circular_correctness_multichoice, ensemble_correctness_multichoice


basedir = "bench_results/emotion_benchmark_split/"
table_outpath = "eval_results/table/emotion_benchmark_split.tex"
table_outpath_percat = "eval_results/table/emotion_benchmark_split_percategory.tex"


# Emotion benchmark split
def structure_check(df):
    task, label, _ = df.name
    n_expected_rows = 7
    assert df.shape[0] == n_expected_rows, df
    assert df.expected.nunique() == len(df.expected), df

if __name__ == "__main__":
    l_stats = []
    l_ensemble = []
    l_stats_percategory = []
    l_ensemble_percategory = []
    for filename in tqdm(os.listdir(basedir)):
        model_name = filename.split("-")[0]
        if model_name not in model_names:
            print(f"Skipping {model_name}")
            continue

        df_responses = pd.read_csv(f"{basedir}/{filename}")
        df_responses = df_responses.rename({
            "circular_index": "circular_id",
            "choice_list": "l_choices",
        }, axis=1)
        df_responses["model_name"] = model_name

        df_responses.groupby(["category", "label", "circular_id"]).apply(structure_check, include_groups=False)

        df_responses["task"] = "emotion-circ-acc"
        df_circ_aggregated = df_responses.groupby(["model_name", "task", "category", "label", "circular_id"]).apply(
            circular_correctness_multichoice, include_groups=False
        )
        df_label_aggregated = df_circ_aggregated.groupby(["model_name", "task", "category", "label"]).apply(np.mean, include_groups=False)
        df_category_aggregated = df_label_aggregated.groupby(["model_name", "task", "category"]).apply(np.mean, include_groups=False)
        df_task_aggregated = df_category_aggregated.groupby(["model_name", "task"]).apply(np.mean, include_groups=False)
        l_stats.append(df_task_aggregated.copy())
        l_stats_percategory.append(df_category_aggregated.copy())

        df_responses["task"] = "emotion-ens-acc"
        df_circ_aggregated = df_responses.groupby(["model_name", "task", "category", "label", "circular_id"]).apply(
            ensemble_correctness_multichoice, include_groups=False
        )
        df_label_aggregated = df_circ_aggregated.groupby(["model_name", "task", "category", "label"]).apply(np.mean, include_groups=False)
        df_category_aggregated = df_label_aggregated.groupby(["model_name", "task", "category"]).apply(np.mean, include_groups=False)
        df_task_aggregated = df_category_aggregated.groupby(["model_name", "task"]).apply(np.mean, include_groups=False)
        l_ensemble.append(df_task_aggregated.copy())
        l_ensemble_percategory.append(df_category_aggregated.copy())


    df = pd.concat([
        pd.concat(l_stats).unstack("model_name"),
        pd.concat(l_ensemble).unstack("model_name"),
    ])

    table = get_latex_table(df)
    table = f"% Emotion benchmark split\n{table}\n\n"
    print(table)
    with open(table_outpath, "wt") as f:
        f.write(table)

    df = pd.concat([
        pd.concat(l_stats_percategory).unstack("model_name"),
        pd.concat(l_ensemble_percategory).unstack("model_name"),
    ])

    table = get_latex_table(df)
    table = f"% Emotion benchmark split, per category\n{table}\n\n"
    print(table)
    with open(table_outpath_percat, "wt") as f:
        f.write(table)
