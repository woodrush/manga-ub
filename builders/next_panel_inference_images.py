import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

from utils import BoundingBox, arrange_images_right_to_left_without_blank
from configs import manga109_base_dir


panels_basedir = "build/next_panel_inference/panels"

img_outdir_combined_base = "tasks/images/next_panel_inference"
csv_outdir_base = "tasks/images/next_panel_inference/labels"

combinations_csv_basepath = "build/next_panel_inference/combinations"
combination_dataset_splits = ["train", "valid", "test"]


def process_row(item):
    _, row = item

    l_ret_impaths = []
    for cropped_type, image_order_type in [
        ("raw", "rightfirst"),
        ("speech_cropped", "rightfirst"),
        ("raw", "leftfirst"),
    ]:
        in_impath_format = panels_basedir + "/" + cropped_type + "/{panel_id}.png"

        collage_function = {
            "rightfirst": lambda x, y: arrange_images_right_to_left_without_blank(x, y, is_rightfirst=True),
            "leftfirst": lambda x, y: arrange_images_right_to_left_without_blank(x, y, is_rightfirst=False),
        }[image_order_type]

        out_impath_basedir = img_outdir_combined_base + "/" + row.split + "/" + cropped_type + "/" + image_order_type
        out_impath = f"{out_impath_basedir}/{row.combination_id}.png"

        l_images_top = [
            Image.open(in_impath_format.format(panel_id=panel_id))
            for panel_id in [
                row.panel_1,
                row.panel_2,
                row.panel_3,
            ]
        ]

        l_images_wrong = [
            Image.open(in_impath_format.format(panel_id=panel_id))
            for panel_id in [
                row.wrong_1,
                row.wrong_2,
            ]
        ]

        im_correct = Image.open(in_impath_format.format(panel_id=row.panel_4))

        l_images_bottom = l_images_wrong
        l_images_bottom.insert(row.i_correct, im_correct)

        ret = collage_function(l_images_top, l_images_bottom)
        ret.save(out_impath)

        l_ret_impaths.append({
            "combination_id": row.combination_id,
            "cropped_type": cropped_type,
            "image_order_type": image_order_type,
            "impath": out_impath,
        })
    df_ret = pd.DataFrame(l_ret_impaths)
    return df_ret


if __name__ == "__main__":
    os.makedirs(csv_outdir_base, exist_ok=True)

    for split in combination_dataset_splits:
        print(f"{split} split")
        os.makedirs(f"{img_outdir_combined_base}/{split}/raw/rightfirst", exist_ok=True)
        os.makedirs(f"{img_outdir_combined_base}/{split}/speech_cropped/rightfirst", exist_ok=True)
        os.makedirs(f"{img_outdir_combined_base}/{split}/raw/leftfirst", exist_ok=True)
        os.makedirs(f"{img_outdir_combined_base}/{split}/speech_cropped/leftfirst", exist_ok=True)

        # Prepare df_combinations
        df_combinations = pd.read_csv(f"{combinations_csv_basepath}/{split}.csv")
        df_combinations["split"] = split

        with Pool(processes=8) as pool:
            ret = pool.map(process_row, tqdm(df_combinations.iterrows(), total=df_combinations.shape[0]))

        df_impaths = pd.concat(ret)
        df_impaths_outpath = f"{csv_outdir_base}/{split}.csv"
        df_impaths.to_csv(df_impaths_outpath, index=None)
