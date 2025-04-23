import pandas as pd
import numpy as np
import sys
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool

from configs import manga109_base_dir, manga_benchmark_dataset_base_dir, panel_id_to_row_csvpath


csv_outpath_basedir = f"build/next_panel_inference/combinations"

four_panel_id_csv_path = "build/four_panel_splits.csv"
panel_id_to_row_csv_path = "build/panel_id_to_row.csv"
panel_id_to_speech_text_csv_path = "build/panel_id_to_speech_text.csv"


if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)

    os.makedirs(csv_outpath_basedir, exist_ok=True)

    df_panel_id_to_row = pd.read_csv(panel_id_to_row_csv_path)
    df_panel_id_to_speech_text = pd.read_csv(panel_id_to_speech_text_csv_path)
    d_panel_id_to_speech_text = dict(zip(
        df_panel_id_to_speech_text.panel_id,
        df_panel_id_to_speech_text.text
    ))

    df_four = pd.read_csv(four_panel_id_csv_path)
    df_four = df_four.sort_values(by="panel_1").reset_index(drop=True)

    # Make panel combinations for each dataset split
    for split in tqdm(df_four.split.unique()):
        df_split = df_four[df_four.split==split]

        # Prepare easy and difficult panel combinations
        l_data = []
        for title in df_split.title.unique():
            df_title = df_split[df_split.title==title]
            df_title_complement = df_split[~(df_split.title==title)]
            l_wrong_panels_easy = (
                list(df_title_complement.panel_1)
                + list(df_title_complement.panel_2)
                + list(df_title_complement.panel_3)
                + list(df_title_complement.panel_4)
            )
            for i_row, row_base in tqdm(df_title.iterrows()):
                row = row_base
                df_difficult = df_title[df_title.sequence_id != row.sequence_id]
                l_wrong_panels_difficult = (
                    list(df_difficult.panel_1)
                    + list(df_difficult.panel_2)
                    + list(df_difficult.panel_3)
                    + list(df_difficult.panel_4)
                )

                # Easy data
                row = row_base.copy()
                wrong_choices_easy = rng.choice(l_wrong_panels_easy, size=2, replace=False)
                i_correct = rng.integers(3)
                row["combination_id"] = f"next_panel_easy_{i_row:05d}"
                row["wrong_1"] = wrong_choices_easy[0]
                row["wrong_2"] = wrong_choices_easy[1]
                row["i_correct"] = i_correct
                row["category"] = "easy"
                row["text_panel_1"] = d_panel_id_to_speech_text[row.panel_1]
                row["text_panel_2"] = d_panel_id_to_speech_text[row.panel_2]
                row["text_panel_3"] = d_panel_id_to_speech_text[row.panel_3]
                row["text_panel_4"] = d_panel_id_to_speech_text[row.panel_4]
                row["text_wrong_1"] = d_panel_id_to_speech_text[row.wrong_1]
                row["text_wrong_2"] = d_panel_id_to_speech_text[row.wrong_2]

                l_data.append(row)

                # Difficult data
                row = row_base.copy()
                wrong_choices_difficult = rng.choice(l_wrong_panels_difficult, size=2, replace=False)
                i_correct = rng.integers(3)
                row["combination_id"] = f"next_panel_difficult_{i_row:05d}"
                row["wrong_1"] = wrong_choices_difficult[0]
                row["wrong_2"] = wrong_choices_difficult[1]
                row["i_correct"] = i_correct
                row["category"] = "difficult"
                row["text_panel_1"] = d_panel_id_to_speech_text[row.panel_1]
                row["text_panel_2"] = d_panel_id_to_speech_text[row.panel_2]
                row["text_panel_3"] = d_panel_id_to_speech_text[row.panel_3]
                row["text_panel_4"] = d_panel_id_to_speech_text[row.panel_4]
                row["text_wrong_1"] = d_panel_id_to_speech_text[row.wrong_1]
                row["text_wrong_2"] = d_panel_id_to_speech_text[row.wrong_2]

                l_data.append(row)

        df_combinations = pd.DataFrame(l_data)
        df_combinations = df_combinations[
            [
                "combination_id",
                "category",
                'sequence_id',
                'panel_1', 'panel_2', 'panel_3', 'panel_4',
                'title', 'i_page',
                "wrong_1",
                "wrong_2",
                "i_correct",
                "text_panel_1",
                "text_panel_2",
                "text_panel_3",
                "text_panel_4",
                "text_wrong_1",
                "text_wrong_2",
            ]
        ]
        df_combinations = df_combinations.sort_values(by=["category", "sequence_id"])


        # Assert properties for df_combinations
        def process_row_assertion(item):
            _, row = item

            l_correct = (
                row.panel_1,
                row.panel_2,
                row.panel_3,
                row.panel_4,
            )
            l_title_correct = list(df_panel_id_to_row[df_panel_id_to_row.panel_id.isin(l_correct)].title)
            assert len(np.unique(l_correct)) == 4
            assert len(np.unique(l_title_correct)) == 1
            title_correct = l_title_correct[0]
            assert title_correct == row.title

            l_title_wrong_1 = list(df_panel_id_to_row[df_panel_id_to_row.panel_id==row.wrong_1].title)
            l_title_wrong_2 = list(df_panel_id_to_row[df_panel_id_to_row.panel_id==row.wrong_2].title)
            assert len(l_title_wrong_1) == len(l_title_wrong_2) == 1
            l_title_wrong = [l_title_wrong_1[0], l_title_wrong_2[0]]

            if row.category == "difficult":
                for title in l_title_wrong:
                    assert title == row.title
            elif row.category == "easy":
                for title in l_title_wrong:
                    assert title != row.title
            else:
                raise ValueError

        with Pool(processes=8) as pool:
            pool.map(process_row_assertion, tqdm(df_combinations.iterrows(), total=df_combinations.shape[0]))
        
        outpath = f"{csv_outpath_basedir}/{split}.csv"
        df_combinations.to_csv(outpath, index=None)
