import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import precision_recall_fscore_support

from utils import model_names, get_latex_table, circular_correctness_multichoice


basedir = "bench_results/next_panel_inference/"
table_outpath = "eval_results/table/next_panel_inference.tex"
table_outpath_detailed = "eval_results/table/next_panel_inference_detailed.tex"


def structure_check(df):
    task, _, label, _ = df.name
    n_expected_rows = 3
    assert df.shape[0] == n_expected_rows, df
    assert df.expected.nunique() == len(df.expected), df


if __name__ == "__main__":
    l_stats = []
    l_stats_details = []
    for i_filename, filename in enumerate(tqdm(os.listdir(basedir))):
        model_name = filename.split("-")[0]

        df_responses = pd.read_csv(f"{basedir}/{filename}")
        df_responses = df_responses.rename({
            "expected_text": "label",
        }, axis=1)
        df_responses["model_name"] = f"{model_name}_{i_filename}"

        df_responses.groupby(["category", "is_with_transcription", "label", "circular_id"]).apply(structure_check, include_groups=False)

        df_circ_aggregated = df_responses.groupby(["model_name", "category", "is_with_transcription", "cropped_type", "image_order_type", "label", "circular_id"]).apply(
            circular_correctness_multichoice, include_groups=False
        )
        df_label_aggregated = df_circ_aggregated.groupby(["model_name", "category", "is_with_transcription", "cropped_type", "image_order_type", "label"]).apply(np.mean, include_groups=False)
        df_task_aggregated = df_label_aggregated.groupby(["model_name", "category", "is_with_transcription", "cropped_type", "image_order_type"]).apply(np.mean, include_groups=False)
        df_task_aggregated_2 = df_task_aggregated.groupby(["model_name", "category", "is_with_transcription", "cropped_type"]).apply(np.mean, include_groups=False)
        l_stats.append(df_task_aggregated_2)
        l_stats_details.append(df_task_aggregated)


    df_stats = pd.concat(l_stats).unstack("model_name")

    table = get_latex_table(df_stats)
    table = f"% Next panel inference\n{table}\n\n"
    print(table)
    with open(table_outpath, "wt") as f:
        f.write(table)

    df_stats_details = pd.concat(l_stats_details).unstack("model_name")

    table = get_latex_table(df_stats_details)
    table = f"% Next panel inference (detailed) \n{table}\n\n"
    print(table)
    with open(table_outpath_detailed, "wt") as f:
        f.write(table)
