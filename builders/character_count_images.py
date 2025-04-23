import os
import pandas as pd
import manga109api
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool

from configs import manga109_base_dir, manga_benchmark_dataset_base_dir
from character_count import load_character_count_dataframe, img_basedir

#==============================================================================
# Dataset path configs
#==============================================================================
img_load_basedir = f"{manga109_base_dir}/images"


#==============================================================================
# Data preparation
#==============================================================================
def process_row(args):
    i_row, row = args
    input_path = f"{img_load_basedir}/{row.title}/{int(row.i_page):03d}.jpg"
    bounding_box = (row.xmin, row.ymin, row.xmax, row.ymax)

    imfilename = f"{row.title}__{row.i_page}__{row.panel_id}.png"
    impath = f"{img_basedir}/{imfilename}"

    image = Image.open(input_path)
    cropped_image = image.crop(bounding_box)
    cropped_image.save(impath)


if __name__ == "__main__":
    df_annotations = load_character_count_dataframe()
    os.makedirs(img_basedir, exist_ok=True)

    print("Obtaining bbox info")
    manga109 = manga109api.Parser(root_dir=manga109_base_dir)

    p = Pool(8)
    p.map(process_row, tqdm(df_annotations.iterrows(), total=df_annotations.shape[0]))
