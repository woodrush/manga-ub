import pandas as pd
import numpy as np
import sys
import os
from PIL import Image
from tqdm import tqdm

from multiprocessing import Pool
from utils import arrange_images_right_to_left_without_blank


csv_outdir = "build"
csv_datapath = f"{csv_outdir}/prepare_panel_localization.csv"


def process_row(item):
    _, row = item
    l_imnames = row.imnames.split(",")
    impaths = [f"{row.im_basedir}/{x}" for x in l_imnames]

    assert l_imnames.index(row.imname_positive) == row.i_positive

    l_images = [Image.open(x) for x in impaths]
    l_top = l_images[:3]
    l_bottom = l_images[3:]

    ret = arrange_images_right_to_left_without_blank(l_top, l_bottom)
    ret.save(row.impath)

if __name__ == "__main__":
    df_data = pd.read_csv(csv_datapath)

    img_out_basedir = os.path.dirname(df_data.impath[0])
    os.makedirs(img_out_basedir, exist_ok=True)

    with Pool(processes=8) as pool:
        pool.map(process_row, tqdm(df_data.iterrows(), total=df_data.shape[0]))
