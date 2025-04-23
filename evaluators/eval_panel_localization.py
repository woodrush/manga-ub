import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import precision_recall_fscore_support

from utils import model_names, get_latex_table, circular_correctness_multichoice, ensemble_correctness_multichoice


basedir = "bench_results/panel_localization/"
table_outpath = "eval_results/table/panel_localization.tex"


def structure_check(df):
    task, label, _ = df.name
    n_expected_rows = 6
    assert df.shape[0] == n_expected_rows, df
    assert df.expected.nunique() == len(df.expected), df

if __name__ == "__main__":
    l_stats = []
    l_ensemble = []
    for filename in tqdm(os.listdir(basedir)):
        model_name = filename.split("-")[0]
        if model_name not in model_names:
            print(f"Skipping {model_name}")
            continue

        df_responses = pd.read_csv(f"{basedir}/{filename}")
        df_responses = df_responses.rename({
            "circular_index": "circular_id",
            "expected_text": "label",
        }, axis=1)
        df_responses["model_name"] = model_name

        df_responses.groupby(["category", "label", "circular_id"]).apply(structure_check, include_groups=False)

        df_responses["task"] = "panel-localization-circ-acc"
        df_circ_aggregated = df_responses.groupby(["model_name", "task", "positive_label", "label", "circular_id"]).apply(
            circular_correctness_multichoice, include_groups=False
        )
        df = df_circ_aggregated.groupby(["model_name", "task", "positive_label", "label"]).apply(np.mean, include_groups=False)
        df = df.groupby(["model_name", "task", "positive_label"]).apply(np.mean, include_groups=False)
        df = df.groupby(["model_name", "task"]).apply(np.mean, include_groups=False)
        df_task_aggregated = df.groupby(["model_name", "task"]).apply(np.mean, include_groups=False)
        l_stats.append(df_task_aggregated)

        df_responses["task"] = "panel-localization-ens-acc"
        df_circ_aggregated = df_responses.groupby(["model_name", "task", "positive_label", "label", "circular_id"]).apply(
            ensemble_correctness_multichoice, include_groups=False
        )
        df = df_circ_aggregated.groupby(["model_name", "task", "positive_label", "label"]).apply(np.mean, include_groups=False)
        df = df.groupby(["model_name", "task", "positive_label"]).apply(np.mean, include_groups=False)
        df = df.groupby(["model_name", "task"]).apply(np.mean, include_groups=False)
        df_task_aggregated = df.groupby(["model_name", "task"]).apply(np.mean, include_groups=False)

        l_ensemble.append(df_task_aggregated)


    df_stats = pd.concat([
        pd.concat(l_stats).unstack("model_name"),
        pd.concat(l_ensemble).unstack("model_name")
    ])

    table = get_latex_table(df_stats)
    table = f"% Panel localization\n{table}\n\n"
    print(table)
    with open(table_outpath, "wt") as f:
        f.write(table)
