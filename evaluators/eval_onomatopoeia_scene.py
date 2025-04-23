import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import model_names, get_latex_table, circular_correctness_multichoice


basedir = "bench_results/onomatopoeia_scene/"
table_outpath = "eval_results/table/onomatopoeia_scene.tex"


def structure_check(df):
    task, _, label, _ = df.name
    n_expected_rows = 3
    assert df.shape[0] == n_expected_rows, df
    assert df.expected.nunique() == len(df.expected), df


if __name__ == "__main__":
    l_stats = []
    for filename in tqdm(os.listdir(basedir)):
        model_name = filename.split("-")[0]

        df_responses = pd.read_csv(f"{basedir}/{filename}")
        df_responses = df_responses.rename({
            "text": "label",
        }, axis=1)
        df_responses["model_name"] = model_name

        df_responses.groupby(["category", "is_transcription_shown", "label", "circular_id"]).apply(structure_check, include_groups=False)

        df_circ_aggregated = df_responses.groupby(["model_name", "category", "is_transcription_shown", "label", "circular_id"]).apply(
            circular_correctness_multichoice, include_groups=False
        )
        df_label_aggregated = df_circ_aggregated.groupby(["model_name", "category", "is_transcription_shown", "label"]).apply(np.mean, include_groups=False)
        df_task_aggregated = df_label_aggregated.groupby(["model_name", "category", "is_transcription_shown"]).apply(np.mean, include_groups=False)
        l_stats.append(df_task_aggregated)


    df_stats = pd.concat(l_stats).unstack("model_name")
    table = get_latex_table(df_stats)
    table = f"% Onomatopoeia scene\n{table}\n\n"
    print(table)
    with open(table_outpath, "wt") as f:
        f.write(table)
