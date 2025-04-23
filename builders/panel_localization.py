import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from utils import make_choice_string


csv_datapath = f"build/prepare_panel_localization.csv"

csv_outdir = "tasks"
prompt_csv_outpath = f"{csv_outdir}/panel_localization.csv"

l_labels_base = [
    "Top right",
    "Top middle",
    "Top left",
    "Bottom right",
    "Bottom middle",
    "Bottom left",
]

prompt_format_loc = """Which of these manga panels shows an {positive} scene?
{choices}
Answer with the option's letter from the given choices directly."""

prompt_format_weather = """Which of these manga panels shows a scene on a {positive} day?
{choices}
Answer with the option's letter from the given choices directly."""

prompt_format_timeofday = """Which of these manga panels shows a scene during the {positive}?
{choices}
Answer with the option's letter from the given choices directly."""

d_prompt_format = {
    "location": prompt_format_loc,
    "time_of_day": prompt_format_timeofday,
    "weather": prompt_format_weather,
}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    os.makedirs(csv_outdir, exist_ok=True)

    df_data = pd.read_csv(csv_datapath).reset_index(drop=True)

    l_prompts = []
    for i_row, row_base in tqdm(df_data.iterrows(), total=df_data.shape[0]):
        expected_text = l_labels_base[row_base.i_positive]

        l_choices_rand = list(rng.permutation(l_labels_base))
        i_correct_base = l_choices_rand.index(expected_text)
        for i_displacement in range(len(l_choices_rand)):
            row = row_base.copy()

            s_choices, l_choices, alphabet_expected = make_choice_string(i_displacement, l_choices_rand, i_correct_base=i_correct_base)

            positive_label = row.positive_label.lower()
            prompt = d_prompt_format[row.category.lower()].format(
                positive=positive_label,
                choices=s_choices,
            )
            row["prompt_id"] = f"panel_loc_{row.image_id}_d{i_displacement}"
            row["circular_id"] = f"panel_loc_{row.image_id}"
            row["i_displacement"] = i_displacement
            row["expected_text"] = expected_text
            row["l_choices"] = ",".join(l_choices)
            row["expected"] = alphabet_expected
            row["prompt"] = prompt
            l_prompts.append(row)

    df_prompts = pd.DataFrame(l_prompts).reset_index(drop=True)
    df_prompts.to_csv(prompt_csv_outpath, index=False)
