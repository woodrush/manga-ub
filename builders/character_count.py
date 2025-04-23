import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from PIL import Image

from configs import manga109_base_dir, manga_benchmark_dataset_base_dir, panel_id_to_row_csvpath

#==============================================================================
# Prompt settings
#==============================================================================
prompt_format = "How many manga characters are visible in this manga panel? Avoid full sentences and answer in one word. Provide the answer as a single number written in arabic numerals."


#==============================================================================
# Dataset path configs
#==============================================================================
datapath = f"{manga_benchmark_dataset_base_dir}/annotations/character_count.csv"
genre_csvpath = f"{manga_benchmark_dataset_base_dir}/annotations/genre.csv"

base_savedir = "tasks"
img_basedir = f"{base_savedir}/images/character_count"
prompt_csv_path = f"{base_savedir}/character_count.csv"


#==============================================================================
# Data preparation
#==============================================================================
def load_character_count_dataframe():
    # Merge dataframes and collect the required data
    df_annotations_base = pd.read_csv(datapath)
    df_panel_id_to_row = pd.read_csv(panel_id_to_row_csvpath)
    df_genre = pd.read_csv(genre_csvpath)

    df_annotations = pd.merge(df_annotations_base, df_panel_id_to_row, on="panel_id", how="left")
    df_annotations = pd.merge(df_annotations, df_genre, on="title", how="left")
    df_annotations = df_annotations.sort_values(by="panel_id").reset_index(drop=True)
    assert df_annotations.shape[0] == df_annotations_base.shape[0]

    return df_annotations

#==============================================================================
# Main
#==============================================================================
if __name__ == "__main__":
    os.makedirs(base_savedir, exist_ok=True)

    df_annotations = load_character_count_dataframe()

    # Make the task data
    l_task_prompts = []

    for i_row, row in tqdm(df_annotations.iterrows(), total=df_annotations.shape[0]):
        expected = "ABCDEFG"[row.n_characters]

        imfilename = f"{row.title}__{row.i_page}__{row.panel_id}.png"
        impath = f"{img_basedir}/{imfilename}"

        r = row.copy()
        r["prompt_id"] = f"character_count_{i_row:05d}"
        r["task"] = "character_count"
        r["label"] = row.n_characters
        r["expected"] = row.n_characters
        r["impath"] = impath
        r["prompt"] = prompt_format
        l_task_prompts.append(r)

    df_prompts = pd.DataFrame(l_task_prompts).reset_index(drop=True)
    df_prompts = df_prompts[[
        'prompt_id', 'task',
        'title', 'i_page', 'genre', 'panel_id', 'label',
        'impath', 'expected', 'prompt',
    ]]
    print(df_prompts)

    assert len(np.unique(df_prompts.prompt_id)) == df_prompts.shape[0]
    df_prompts.to_csv(prompt_csv_path, index=False)
