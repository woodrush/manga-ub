import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool

from configs import manga109_base_dir, manga_benchmark_dataset_base_dir
from recognition_background import load_background_recognition_dataframe

#==============================================================================
# Dataset path configs
#==============================================================================
base_savedir = "tasks/images/recognition_background"
img_load_basedir = f"{manga109_base_dir}/images"


#==============================================================================
# Data preparation
#==============================================================================
def process_row(args):
    i_row, row = args
    input_path = f"{img_load_basedir}/{row.title}/{int(row.i_page):03d}.jpg"
    bounding_box = (row.xmin, row.ymin, row.xmax, row.ymax)

    image = Image.open(input_path)
    cropped_image = image.crop(bounding_box)
    cropped_image.save(row.impath)

if __name__ == "__main__":
    df_annotations = load_background_recognition_dataframe()
    os.makedirs(base_savedir, exist_ok=True)

    p = Pool(8)
    result = p.map(process_row, tqdm(df_annotations.iterrows(), total=df_annotations.shape[0]))
