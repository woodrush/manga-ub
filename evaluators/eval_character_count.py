import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import model_names, get_latex_table


basedir = "bench_results/character_count/"
table_outpath = "eval_results/table/character_count.tex"
table_outpath_perlabel = "eval_results/table/character_count_perlabel.tex"


def parse_integer_response(s):
    if type(s) == str:
        s = s.strip()
    try:
        n = int(s)
        return n
    except:
        return "n/a"

def charcount_correctness(df):
    assert df.shape[0] == 1
    expected = int(df.expected.iloc[0])
    response = parse_integer_response(df.response.iloc[0])
    is_correct = expected == response
    return 1 if is_correct else 0


if __name__ == "__main__":
    os.makedirs("eval_results/table", exist_ok=True)

    l_stats = []
    l_per_label = []
    for filename in tqdm(os.listdir(basedir)):
        model_name = filename.split("-")[0]
        if model_name not in model_names:
            print(f"Skipping {model_name}")
            continue

        df_responses = pd.read_csv(f"{basedir}/{filename}")
        df_responses["model_name"] = model_name

        df_circ_aggregated = df_responses.groupby(["model_name", "task", "label", "prompt_id"]).apply(charcount_correctness, include_groups=False)
        df_label_aggregated = df_circ_aggregated.groupby(["model_name", "task", "label"]).apply(np.mean, include_groups=False)
        df_task_aggregated = df_label_aggregated.groupby(["model_name", "task"]).apply(np.mean, include_groups=False)
        l_stats.append(df_task_aggregated)
        l_per_label.append(df_label_aggregated)

    df_stats = pd.concat(l_stats).unstack("model_name")

    table = get_latex_table(df_stats)
    table = f"% Character count\n{table}\n\n"
    print(table)
    with open(table_outpath, "wt") as f:
        f.write(table)

    df_stats = pd.concat(l_per_label).unstack("model_name")
    df_stats.loc["mean"] = df_stats.mean()

    table = get_latex_table(df_stats)
    table = f"% Character count, per-label\n{table}\n\n"
    print(table)
    with open(table_outpath_perlabel, "wt") as f:
        f.write(table)
