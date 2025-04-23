import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm

from recognition_background import load_background_recognition_dataframe
from utils import chunk_generator

csv_outdir = "build"
csv_outpath = f"{csv_outdir}/prepare_panel_localization.csv"
image_out_basedir = "tasks/images/panel_localization"


if __name__ == "__main__":
    os.makedirs(csv_outdir, exist_ok=True)
    os.makedirs(image_out_basedir, exist_ok=True)

    rng = np.random.default_rng(seed=0)

    df_bg = load_background_recognition_dataframe()

    l_image_ingredients = []
    i_displacement = 0
    i_image = 0
    for category in df_bg.category.unique():
        df = df_bg[df_bg.category==category]

        # Avoid the label "Snowy" from the data
        labels = [x for x in df.label.unique() if x != "Snowy"]

        for positive_label in labels:
            l_negatives = [x for x in labels if x != positive_label]
            assert len(l_negatives) == 1
            negative_label = l_negatives[0]

            # Use RNG
            df_positive = df[df.label==positive_label].sample(frac=1,random_state=rng)
            df_negative = df[df.label==negative_label].sample(frac=1,random_state=rng)

            assert len(df_positive.impath.unique()) == df_positive.shape[0]
            assert len(df_negative.impath.unique()) == df_negative.shape[0]

            for (_, row_positive), rows_negative in tqdm(zip(
                df_positive.iterrows(),
                chunk_generator(df_negative.iterrows(), chunk_size=5),
            )):
                impaths = [x.impath for _, x in rows_negative]
                impaths.insert(i_displacement, row_positive.impath)
                i_displacement += 1
                i_displacement %= 6

                dirnames = np.unique([os.path.dirname(x) for x in impaths])
                imnames = [os.path.basename(x) for x in impaths]
                assert len(dirnames) == 1
                assert len(np.unique(imnames)) == 6
                assert len(np.unique(impaths)) == len(impaths) == 6
                assert all("," not in x for x in imnames)
                im_basedir = dirnames[0]

                i_positive = impaths.index(row_positive.impath)
                imname_positive = os.path.basename(row_positive.impath)

                image_id = f"im{i_image:05d}"
                impath = f"{image_out_basedir}/{image_id}.png"
                i_image += 1
                l_image_ingredients.append({
                    "category": category,
                    "positive_label": positive_label,
                    "negative_label": negative_label,
                    "image_id": image_id,
                    "i_positive": i_positive,
                    "imnames": ",".join(imnames),
                    "im_basedir": im_basedir,
                    "imname_positive": imname_positive,
                    "impath": impath,
                })
    df_out = pd.DataFrame(l_image_ingredients)
    df_out.to_csv(csv_outpath, index=None)
