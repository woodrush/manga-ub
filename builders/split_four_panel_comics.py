import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm
from multiprocessing import Pool

from configs import manga109_base_dir, manga_benchmark_dataset_base_dir, panel_id_to_row_csvpath


four_panel_id_csv_path =  f"{manga_benchmark_dataset_base_dir}/annotations/four_panel_comics.csv"
panel_id_to_row_csv_path = "build/panel_id_to_row.csv"


if __name__ == "__main__":
    df_panel_id_to_row = pd.read_csv(panel_id_to_row_csv_path)

    # Add title and page info to the four panel comic annotations
    df_four = pd.read_csv(four_panel_id_csv_path)
    df_four = df_four.sort_values(by="panel_1").reset_index(drop=True)
    df_four["panel_id"] = df_four.panel_1
    df_four = pd.merge(df_four, df_panel_id_to_row, on="panel_id", how="left")

    # Split train_valid/test using `r_pages`
    df_four_4titles = df_four[df_four.title!="TetsuSan"]
    r_pages = 0.7

    l_add = []
    for title in df_four_4titles.title.unique():
        df = df_four_4titles[df_four_4titles.title==title]
        l_i_pages = list(sorted(df.i_page.unique()))
        n_pages = int(len(l_i_pages) * r_pages)

        use_pages = l_i_pages[:n_pages]
        df_add = df[df.i_page.isin(use_pages)]
        l_add.append(df_add)

    df_train_valid = pd.concat(l_add)

    df_test_a = df_four_4titles[~df_four_4titles.sequence_id.isin(df_train_valid.sequence_id)]
    df_test = pd.concat([
        df_test_a,
        df_four[df_four.title=="TetsuSan"]
    ])

    # Split train/valid using `r_pages`
    r_pages = 0.9

    l_add = []
    for title in df_train_valid.title.unique():
        df = df_train_valid[df_train_valid.title==title]
        l_i_pages = list(sorted(df.i_page.unique()))
        n_pages = int(len(l_i_pages) * r_pages)

        use_pages = l_i_pages[:n_pages]
        df_add = df[df.i_page.isin(use_pages)]
        l_add.append(df_add)

    df_train = pd.concat(l_add)

    df_valid = df_train_valid[~df_train_valid.sequence_id.isin(df_train.sequence_id)]

    print(df_train.title.value_counts(), df_train.title.value_counts().sum())
    print(df_valid.title.value_counts(), df_valid.title.value_counts().sum())
    print(df_test.title.value_counts(), df_test.title.value_counts().sum())

    df_out = df_four.copy()
    df_out["split"] = None
    for split_label, df in [
        ("train", df_train),
        ("test", df_test),
        ("valid", df_valid),
    ]:
        sequence_ids_in_b = df['sequence_id'].tolist()

        df_out.loc[df_out['sequence_id'].isin(sequence_ids_in_b), 'split'] = split_label
    df_out = df_out[[
        "split",
        "sequence_id",
        "panel_1",
        "panel_2",
        "panel_3",
        "panel_4",
        "title",
        "i_page",
    ]]
    df_out = df_out.sort_values(by=["split", "sequence_id"])
    df_out.to_csv("build/four_panel_splits.csv", index=None)

    assert df_out[df_out.split=="train"].shape[0] == df_train.shape[0]
    assert df_out[df_out.split=="valid"].shape[0] == df_valid.shape[0]
    assert df_out[df_out.split=="test"].shape[0] == df_test.shape[0]
