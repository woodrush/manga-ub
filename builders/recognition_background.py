import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from PIL import Image

from configs import manga109_base_dir, manga_benchmark_dataset_base_dir, panel_id_to_row_csvpath
from utils import make_choice_string


#==============================================================================
# Prompt settings
#==============================================================================
d_prompt_base_question = {
    "Location":            "Which location is shown in this manga panel?",
    "Time_of_day":         "Which time of day is shown in this manga panel?",
    "Weather":             "Which weather is shown in this manga panel?",
    "Weather_difficult":   "Which weather is shown in this manga panel?",
}

d_base_choices = {
    "Location":            ['Indoors', 'Outdoors'],
    "Time_of_day":         ['Day', 'Night'],
    "Weather":             ['Sunny', 'Rainy'],
    "Weather_difficult":   ['Sunny', 'Rainy', 'Snowy'],
}

prompt_format = """{base_question}
{s_choices}
Answer with the option's letter from the given choices directly."""

#==============================================================================
# Dataset path configs
#==============================================================================
datapath = f"{manga_benchmark_dataset_base_dir}/annotations/recognition_background.csv"
genre_csvpath = f"{manga_benchmark_dataset_base_dir}/annotations/genre.csv"

base_savedir = "tasks"
img_load_basedir = f"{manga109_base_dir}/images"
img_savedir = f"{base_savedir}/images/recognition_background"
prompt_csv_path = f"{base_savedir}/recognition_background.csv"


#==============================================================================
# Data preparation
#==============================================================================
def load_background_recognition_dataframe():
    # Merge dataframes and collect the required data
    df_annotations_base = pd.read_csv(datapath)
    df_panel_id_to_row = pd.read_csv(panel_id_to_row_csvpath)
    df_genre = pd.read_csv(genre_csvpath)

    df_annotations = pd.merge(df_annotations_base, df_panel_id_to_row, on="panel_id", how="left")
    df_annotations = pd.merge(df_annotations, df_genre, on="title", how="left")
    df_annotations = df_annotations.sort_values(by="panel_id").reset_index(drop=True)
    assert df_annotations.shape[0] == df_annotations_base.shape[0]

    df_annotations["impath"] = None
    for i_row, row in tqdm(df_annotations.iterrows(), total=df_annotations.shape[0]):
        img_savefilename = f"{row.title}__{row.i_page}__{row.panel_id}.png"
        output_impath = f"{img_savedir}/{img_savefilename}"
        df_annotations.at[i_row, "impath"] = output_impath

    assert None not in list(df_annotations.impath)

    return df_annotations


#==============================================================================
# Main
#==============================================================================
if __name__ == "__main__":
    # Specify random seed
    rng = np.random.default_rng(0)

    os.makedirs("build", exist_ok=True)
    os.makedirs(base_savedir, exist_ok=True)

    # Generate base dataset info from the annotations
    df_annotations = load_background_recognition_dataframe()
    print(df_annotations)

    l_task_prompts = []

    # Tasks except Weather-3
    df_dataset_tmp = df_annotations[df_annotations.label != "Snowy"]
    for i_row, row in tqdm(df_dataset_tmp.iterrows(), total=df_dataset_tmp.shape[0]):
        category = row.category
        base_question = d_prompt_base_question[category]
        base_choices = d_base_choices[category]

        l_emotions_rand = rng.permutation(base_choices)
        negative_choices = [x for x in l_emotions_rand if x != row.label]
        base_choices = [row.label] + negative_choices

        circular_index = f"{category}_im{i_row:05d}"

        for i_displacement in range(len(base_choices)):
            s_choices, l_choices, expected = make_choice_string(i_displacement, base_choices)
            prompt = prompt_format.format(
                base_question=base_question,
                s_choices=s_choices
            )
            r = row.copy()
            r["prompt_id"] = f"{circular_index}_d{i_displacement:02d}"
            r["task"] = category
            r["prompt"] = prompt
            r["circular_index"] = circular_index
            r["choice_list"] = ",".join(l_choices)
            r["expected"] = expected
            r["impath"] = row.impath
            l_task_prompts.append(r)

    # Weather-3 Task
    df_dataset_tmp = df_annotations[df_annotations.category == "Weather"]
    for i_row, row in tqdm(df_dataset_tmp.iterrows(), total=df_dataset_tmp.shape[0]):
        category = "Weather_difficult"
        base_question = d_prompt_base_question[category]
        base_choices = d_base_choices[category]

        l_emotions_rand = rng.permutation(base_choices)
        negative_choices = [x for x in l_emotions_rand if x != row.label]
        base_choices = [row.label] + negative_choices

        circular_index = f"{category}_im{i_row:05d}"

        for i_displacement in range(len(base_choices)):
            s_choices, l_choices, expected = make_choice_string(i_displacement, base_choices)
            prompt = prompt_format.format(
                base_question=base_question,
                s_choices=s_choices
            )
            r = row.copy()
            r["prompt_id"] = f"{circular_index}_d{i_displacement:02d}"
            r["task"] = category
            r["prompt"] = prompt
            r["circular_index"] = circular_index
            r["choice_list"] = ",".join(l_choices)
            r["expected"] = expected
            r["impath"] = row.impath
            l_task_prompts.append(r)

    df_prompts = pd.DataFrame(l_task_prompts).reset_index(drop=True)
    df_prompts = df_prompts[[
        'prompt_id', 'circular_index',
        'title', 'genre', 'task', 'label', 'impath',
        'choice_list', 'expected', 'prompt',
    ]]
    print(df_prompts)

    df_prompts.to_csv(prompt_csv_path, index=False)

    assert len(np.unique(df_prompts.prompt_id)) == df_prompts.shape[0]
