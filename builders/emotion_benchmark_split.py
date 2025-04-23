import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import calculate_md5, make_choice_string

from configs import kangaiset_base_dir, manga_benchmark_dataset_base_dir

rng = np.random.default_rng(0)

#==============================================================================
# Prompt settings
#==============================================================================
prompt_format = """This is a character from manga. Which emotion does this character show?
{choices}
Answer with the option's letter from the given choices directly."""

l_emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
l_emotions = list(sorted(l_emotions))


#==============================================================================
# Dataset path configs
#==============================================================================
kangaiset_csv_md5_sum_expected = "305f2e74fbb8463ca81d9dcfb0164664"

test_basedir = f"{kangaiset_base_dir}/images/face/all"
test_basedir_body = f"{kangaiset_base_dir}/images/body/all"
test_basedir_panel = f"{kangaiset_base_dir}/images/panel/all"
kangaiset_annotation_short_csv_path = f"{kangaiset_base_dir}/kangaiset_annotation_short.csv"

kangaiset_subset_keys_csv_path = f"{manga_benchmark_dataset_base_dir}/annotations/kangaiset_subset_keys.csv"

task_csv_output_basedir = "tasks"
task_csv_output_path = f"{task_csv_output_basedir}/emotion_benchmark_split.csv"
task_face_body_csv_output_path = f"{task_csv_output_basedir}/emotion_benchmark_split_face_body.csv"

genre_csv_path = f"{manga_benchmark_dataset_base_dir}/annotations/genre.csv"

os.makedirs(task_csv_output_basedir, exist_ok=True)


#==================================================================================================
# Extract the benchmark subset data from KangaiSet using the subset keys
#==================================================================================================
def get_kangaiset_image_path(bb_type, split_type, row):
    assert bb_type in {"face", "body", "panel"}
    assert split_type in {"test_external", "all", "train"}

    genre = row.book_genre
    title = row.manga_name
    emotion = row.emotion_str
    id_character = row.face_character_id
    id_face = row.face_id
    id_body = row.body_id
    id_panel = row.frame_id

    if bb_type == "face":
        id_str = f"{id_character}_{id_face}"
    elif bb_type == "body":
        id_str = f"{id_character}_{id_face}_{id_body}"
    elif bb_type == "panel":
        id_str = f"{id_character}_{id_face}_{id_panel}"

    filename = f"{genre}_{title}_{id_str}.png"
    base_path = f"{kangaiset_base_dir}/images/{bb_type}/{split_type}/{emotion}"
    impath = f"{base_path}/{filename}"

    return impath

kangaiset_csv_md5_sum = calculate_md5(kangaiset_annotation_short_csv_path)
assert kangaiset_csv_md5_sum == kangaiset_csv_md5_sum_expected, f"KangaiSet CSV md5 sum mismatch: Got {kangaiset_csv_md5_sum}"

df_emotion_subset_keys = pd.read_csv(kangaiset_subset_keys_csv_path)
df_kangaiset_annotation_short = pd.read_csv(kangaiset_annotation_short_csv_path)

kangaiset_keys_full = df_kangaiset_annotation_short[df_kangaiset_annotation_short.columns[0]]
subset_keys = df_emotion_subset_keys.kangaiset_key

df_emotion_subset = df_kangaiset_annotation_short[kangaiset_keys_full.isin(subset_keys)].copy()
df_emotion_subset["title"] = df_emotion_subset.manga_name
df_emotion_subset["label"] = df_emotion_subset.emotion_str
df_emotion_subset["id_face_bb"] = df_emotion_subset.face_id

df_emotion_subset["impath_face"] = [get_kangaiset_image_path("face", "all", row) for _, row in df_emotion_subset.iterrows()]
df_emotion_subset["impath_body"] = [get_kangaiset_image_path("body", "all", row) for _, row in df_emotion_subset.iterrows()]
df_emotion_subset["impath_panel"] = [get_kangaiset_image_path("panel", "all", row) for _, row in df_emotion_subset.iterrows()]
df_emotion_subset = df_emotion_subset[['title', 'label', 'id_face_bb', 'impath_face', 'impath_body', 'impath_panel']]

# Sort the values by the face bb keys so that the order becomes consistent
df_emotion_subset = df_emotion_subset.sort_values(by="id_face_bb").reset_index(drop=True)


#==================================================================================================
# Add genre information for evaluation
#==================================================================================================
df_genre = pd.read_csv(genre_csv_path)

d_title_to_genre = dict(zip(df_genre.title, df_genre.genre))

df_emotion_subset["genre"] = df_emotion_subset.title.map(d_title_to_genre.get)


#==================================================================================================
# Generate the prompts
#==================================================================================================
l_data_face = []
l_data_body = []
l_data_panel = []

for i_row, (i_row, row) in tqdm(enumerate(df_emotion_subset.iterrows())):
    # Make sure that the expected answer is A when there is no displacement,
    # without loss of randomness
    l_emotions_rand = rng.permutation(l_emotions)
    negative_choices = [x for x in l_emotions_rand if x.lower() != row.label]
    assert len(negative_choices) == 6

    label = row.label[0].upper() + row.label[1:]
    choices = [label] + negative_choices
    assert len(choices) == 7

    q_index = i_row
    for i_displacement in range(len(choices)):
        s_choices, l_choices, expected = make_choice_string(i_displacement, choices)
        prompt_text = prompt_format.format(choices=s_choices)

        str_l_choices = ",".join(l_choices)

        l_data_face.append({
            "prompt_id": f"face_bb{i_row:05d}_d{i_displacement:02d}",
            "circular_index": f"face_bb{i_row:05d}",
            "task": "emotion",
            "category": "face",
            "label": label,
            "title": row.title,
            "genre": row.genre,
            "choice_list": str_l_choices,
            "expected": expected,
            "prompt": prompt_text,
            "impath": row.impath_face
        })

        l_data_body.append({
            "prompt_id": f"body_bb{i_row:05d}_d{i_displacement:02d}",
            "circular_index": f"body_bb{i_row:05d}",
            "task": "emotion",
            "category": "body",
            "label": label,
            "title": row.title,
            "genre": row.genre,
            "choice_list": str_l_choices,
            "expected": expected,
            "prompt": prompt_text,
            "impath": row.impath_body
        })

df_prompts_grand = pd.DataFrame(
    l_data_face + l_data_body + l_data_panel
).reset_index(drop=True)

df_prompts_grand = df_prompts_grand[[
    "prompt_id",
    "circular_index",
    "category",
    "task",
    "label",
    "title",
    "genre",
    "impath",
    "choice_list",
    "expected",
    "prompt",
]]

df_prompts_grand.to_csv(task_csv_output_path)
