import os
import sys
import numpy as np
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

from configs import manga_benchmark_dataset_base_dir
from utils import make_choice_string, debug

csv_inpath = "build/onomatopoeia_scene_panels.csv"

prompt_csv_outpath = "tasks/onomatopoeia_scene.csv"

prompt_format = """Which description best fits the scene shown in this manga panel?
{choices}
Answer with the option's letter from the given choices directly."""

prompt_format_withtranscription = """The onomatopoeia {s_onoms} {onom_is_are} shown in this manga panel. Which description best fits the scene shown in this manga panel?
{choices}
Answer with the option's letter from the given choices directly."""


if __name__ == "__main__":
    os.makedirs("tasks", exist_ok=True)

    df_onom_base = pd.read_csv(csv_inpath)

    with open(f"{manga_benchmark_dataset_base_dir}/annotations/onomatopoeia_descriptions.json", "rt") as f:
        d_onom_desc = json.loads(f.read())
    d_descriptions = d_onom_desc["descriptions"]
    d_negative = d_onom_desc["negative"]

    l_prompts = []
    for _, row in tqdm(df_onom_base.iterrows(), total=df_onom_base.shape[0]):
        l_negative = d_negative[row.text]
        l_onoms = [row.text] + l_negative
        l_choices = list(map(d_descriptions.get, l_onoms))
        expected_text = l_choices[0]

        for i_displacement in range(3):
            s_choice, _, expected_alphabet = make_choice_string(i_displacement, l_choices)
            prompt = prompt_format.format(choices=s_choice)

            # Quesion without the transcription within the prompt
            r = row.copy()
            r["prompt_id"] = f"onom_scene_{row.onom_id}_{row.category}_notranscription_d{i_displacement:02d}"
            r["circular_id"] = f"onom_scene_{row.onom_id}_{row.category}_notranscription"
            r["category"] = row.category
            r["is_transcription_shown"] = "False"
            r["choices"] = s_choice
            r["expected_text"] = expected_text
            r["expected"] = expected_alphabet
            r["impath"] = row.impath
            r["prompt"] = prompt
            l_prompts.append(r)

            # Quesion without the transcription within the prompt
            r = r.copy()
            l_contained_onoms = r["contained_onoms"].split(",")
            s_contained_onoms = "".join(f"「{s}」" for s in l_contained_onoms)
            onom_is_are = "is" if len(l_contained_onoms) == 1 else "are"
            prompt = prompt_format_withtranscription.format(
                onom_is_are=onom_is_are,
                s_onoms=s_contained_onoms,
                choices=s_choice,
            )
            r["is_transcription_shown"] = "True"
            r["prompt_id"] = f"onom_scene_{row.onom_id}_{row.category}_withtranscription_d{i_displacement:02d}"
            r["circular_id"] = f"onom_scene_{row.onom_id}_{row.category}_withtranscription"
            r["prompt"] = prompt
            l_prompts.append(r)

    df_out = pd.DataFrame(l_prompts).reset_index(drop=True)
    df_out = df_out[[
        'prompt_id', 'circular_id',
        'category', 'is_transcription_shown',
        'title', 'i_page', 'onom_id',
        'text', 'xmin', 'ymin', 'xmax', 'ymax',
        'choices', 'expected_text', 'expected',
        'impath', 'prompt',
    ]]
    df_out.to_csv(prompt_csv_outpath, index=None)
