import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils import BoundingBox, arrange_images_right_to_left_without_blank, make_choice_string, debug
from configs import manga109_base_dir

prompt_csv_out_basepath = "tasks/next_panel_inference"

combinations_basepath = "build/next_panel_inference/combinations"

image_labeldir_base = "tasks/images/next_panel_inference/labels"
combinations_csv_basepath = "build/next_panel_inference/combinations"
combination_dataset_splits = ["train", "valid", "test"]


d_l_choices = {
    "rightfirst": [
        "Bottom right",
        "Bottom middle",
        "Bottom left",
    ],
    "leftfirst": [
        "Bottom left",
        "Bottom middle",
        "Bottom right",
    ],
}

d_image_order_type_prompt = {
    "rightfirst": "right to left",
    "leftfirst": "left to right",
}

prompt_format = """The top row shows three consecutive manga panels read from {direction}. Which of the three panels on the bottom row appears immediately after the panels on the top row?
{choices}
Answer with the option's letter from the given choices directly."""

prompt_format_with_transcript = """The top row shows three consecutive manga panels read from {direction}. Here are their dialogues:
Panel 1:
{dialogue_panel_1}
Panel 2:
{dialogue_panel_2}
Panel 3:
{dialogue_panel_3}

Here are the dialogues in the panels on the bottom row:
Bottom right:
{dialogue_choice_A}
Bottom middle:
{dialogue_choice_B}
Bottom left:
{dialogue_choice_C}

Which of the three panels on the bottom row appears immediately after the panels on the top row?
{choices}
Answer with the option's letter from the given choices directly."""


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    os.makedirs(prompt_csv_out_basepath, exist_ok=True)

    for split in combination_dataset_splits:
        df_impaths = pd.read_csv(f"{image_labeldir_base}/{split}.csv")
        df_combinations = pd.read_csv(f"{combinations_basepath}/{split}.csv")
        df_data = pd.merge(df_impaths, df_combinations, left_on="combination_id", right_on="combination_id")

        l_ret = []
        for _, row_base in tqdm(df_data.iterrows(), total=df_data.shape[0]):
            cropped_type = row_base.cropped_type
            image_order_type = row_base.image_order_type

            direction = d_image_order_type_prompt[row_base.image_order_type]

            # Note that the interpretation of i_correct depends on image_order_type
            if image_order_type == "rightfirst":
                expected_text = d_l_choices["rightfirst"][row_base.i_correct]
            elif image_order_type == "leftfirst":
                expected_text = d_l_choices["leftfirst"][row_base.i_correct]

            l_labels_base = d_l_choices[image_order_type]
            l_choices_rand = list(rng.permutation(l_labels_base))

            i_correct_base = l_choices_rand.index(expected_text)

            # Prepare the transcriptions
            l_text_choices_base = [
                row_base.text_wrong_1,
                row_base.text_wrong_2,
            ]
            l_text_choices_base.insert(row_base.i_correct, row_base.text_panel_4)
            is_null = lambda x: x == "" or (type(x) != str and np.isnan(x))
            l_text_choices_base = ["(no text)" if is_null(s) else s for s in l_text_choices_base]

            l_text_context = ["(no text)" if is_null(s) else s for s in [
                row_base.text_panel_1,
                row_base.text_panel_2,
                row_base.text_panel_3,
            ]]

            l_prompts = []
            for i_displacement in range(len(l_choices_rand)):
                row = row_base.copy()

                s_choices, l_choices, alphabet_expected = make_choice_string(i_displacement, l_choices_rand, i_correct_base=i_correct_base)

                prompt = prompt_format.format(
                    direction=direction,
                    choices=s_choices,
                )
                row["prompt_id"] = f"{row.combination_id}_{cropped_type}_{image_order_type}_notranscription_d{i_displacement}"
                row["circular_id"] = f"{row.combination_id}_{cropped_type}_{image_order_type}_notranscription"
                row["i_displacement"] = i_displacement
                row["is_with_transcription"] = "False"
                row["l_choices"] = ",".join(l_choices)
                row["expected_text"] = expected_text
                row["expected"] = alphabet_expected
                row["impath"] = row.impath
                row["prompt"] = prompt
                l_prompts.append(row)

                # If it is the base case with (raw, rightfirst), add the ablation for putting the text in the prompt
                is_base_case = row.image_order_type == "rightfirst" and row.cropped_type == "raw"
                if is_base_case:
                    row = row.copy()

                    prompt = prompt_format_with_transcript.format(
                        dialogue_panel_1=l_text_context[0],
                        dialogue_panel_2=l_text_context[1],
                        dialogue_panel_3=l_text_context[2],
                        dialogue_choice_A=l_text_choices_base[0],
                        dialogue_choice_B=l_text_choices_base[1],
                        dialogue_choice_C=l_text_choices_base[2],
                        direction=direction,
                        choices=s_choices,
                    )
                    row["prompt_id"] = f"{row.combination_id}_{cropped_type}_{image_order_type}_withtranscription_d{i_displacement}"
                    row["circular_id"] = f"{row.combination_id}_{cropped_type}_{image_order_type}_withtranscription"
                    row["i_displacement"] = i_displacement
                    row["is_with_transcription"] = "True"
                    row["l_choices"] = ",".join(l_choices)
                    row["expected_text"] = expected_text
                    row["expected"] = alphabet_expected
                    row["impath"] = row.impath
                    row["prompt"] = prompt
                    l_prompts.append(row)

            df_ret = pd.DataFrame(l_prompts)
            l_ret.append(df_ret)

        df_prompts = pd.concat(l_ret).reset_index(drop=True)
        outpath = f"{prompt_csv_out_basepath}/{split}.csv"
        df_prompts.to_csv(outpath, index=False)
