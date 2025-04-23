import os
import numpy as np
import pandas as pd
import manga109api
from PIL import Image, ImageDraw
from tqdm import tqdm
from multiprocessing import Pool

from configs import manga109_base_dir


img_out_basedir_raw = "build/next_panel_inference/panels/raw"
img_out_basedir_speech_cropped = "build/next_panel_inference/panels/speech_cropped"

four_panel_splits_csv_path = "build/four_panel_splits.csv"
panel_id_to_row_csv_path = "build/panel_id_to_row.csv"


if __name__ == "__main__":
    os.makedirs(img_out_basedir_raw, exist_ok=True)
    os.makedirs(img_out_basedir_speech_cropped, exist_ok=True)

    df_four_panels = pd.read_csv(four_panel_splits_csv_path)

    df_panel_id_to_row = pd.read_csv(panel_id_to_row_csv_path)
    df_panel_id_to_row["title_page"] = df_panel_id_to_row.title + "_p" + df_panel_id_to_row.i_page.map(str)

    l_panel_id_unique = pd.concat([
        df_four_panels.panel_1,
        df_four_panels.panel_2,
        df_four_panels.panel_3,
        df_four_panels.panel_4,
    ]).unique()

    df_four_panels = df_panel_id_to_row[df_panel_id_to_row.panel_id.isin(l_panel_id_unique)]

    manga109 = manga109api.Parser(manga109_base_dir)

    for title in df_four_panels.title.unique():
        df_title = df_four_panels[df_four_panels.title==title]

        title_annotations = manga109.get_annotation(book=title)["page"]

        def process_title_page(title_page):
            df_page = df_title[df_title.title_page==title_page]

            row = df_page.iloc[0]
            image_page = Image.open(f"{manga109_base_dir}/images/{row.title}/{row.i_page:03d}.jpg")

            # Write the non-speech-cropped panels
            for _, row in df_page.iterrows():
                image_panel = image_page.crop((
                    row.xmin,
                    row.ymin,
                    row.xmax,
                    row.ymax,
                ))
                image_panel.save(f"{img_out_basedir_raw}/{row.panel_id}.png")

            # Make and write the speech-cropped panels
            draw = ImageDraw.Draw(image_page)

            page_annotations = title_annotations[row.i_page]
            for text_bb in page_annotations["text"]:
                draw.rectangle((
                    text_bb["@xmin"],
                    text_bb["@ymin"],
                    text_bb["@xmax"],
                    text_bb["@ymax"],
                ), fill=(0, 0, 0))

            for _, row in df_page.iterrows():
                image_panel = image_page.crop((
                    row.xmin,
                    row.ymin,
                    row.xmax,
                    row.ymax,
                ))
                image_panel.save(f"{img_out_basedir_speech_cropped}/{row.panel_id}.png")

        with Pool(processes=8) as pool:
            ret = pool.map(process_title_page, tqdm(df_title.title_page.unique()))
