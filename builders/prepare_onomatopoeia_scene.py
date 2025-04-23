import os
import sys
import numpy as np
import pandas as pd
import manga109api

from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
import shapely.geometry

from configs import manga109_base_dir, coo_base_dir, manga_benchmark_dataset_base_dir
from utils import BoundingBox, crop_polygon, debug

csv_outpath = "build/onomatopoeia_scene_panels.csv"
images_outpath = "tasks/images/onomatopoeia_scene"

coo_id_csv_path = f"{manga_benchmark_dataset_base_dir}/annotations/onomatopoeia_COO_ids.csv"


def get_onom_vertices(d_onom):
    ret = []
    for i in range(len(d_onom.keys()) // 2):
        if f"@x{i}" in d_onom.keys() and f"@y{i}" in d_onom.keys():
            x = d_onom[f"@x{i}"]
            y = d_onom[f"@y{i}"]
            v = (x, y)
            ret.append(v)
        else:
            break
    return ret

def onom_annotation_to_str(d_onom):
    l_v = get_onom_vertices(d_onom)
    s = "|".join(f"{x},{y}" for x, y in l_v)
    return s

def get_onom_dataset_by_id(query_onom_ids):
    coo_dataset = manga109api.Parser(root_dir=coo_base_dir)

    print("Loading COO dataset...")
    l_ret = []
    for title in tqdm(coo_dataset.books):
        annotations = coo_dataset.get_annotation(title)["page"]
        for page in annotations:
            if "onomatopoeia" not in page.keys():
                continue
            for bb_onom in page["onomatopoeia"]:
                hit_onom_id = str(bb_onom["@id"])
                if str(hit_onom_id) in query_onom_ids:
                    onom_annotations = onom_annotation_to_str(bb_onom)
                    query_onom_ids.remove(hit_onom_id)
                    l_ret.append({
                        "title": title,
                        "i_page": page["@index"],
                        "onom_id": hit_onom_id,
                        "text": bb_onom["#text"],
                        "polygon": onom_annotations
                    })
                    if len(query_onom_ids) == 0:
                        return l_ret
    raise ValueError

if __name__ == "__main__":
    # Extract the COO dataset annotations
    df_coo_ids = pd.read_csv(coo_id_csv_path)

    set_coo_ids = set(df_coo_ids.onom_id.map(str))
    ret = get_onom_dataset_by_id(set_coo_ids)

    df_onom_dataset = pd.DataFrame(ret)

    # Calculate the bounding boxes for each annotation
    manga109 = manga109api.Parser(root_dir=manga109_base_dir)
    coo_dataset = manga109api.Parser(root_dir=coo_base_dir)


    l_ret = []
    for title in df_onom_dataset.title.unique():
        annotations = manga109.get_annotation(book=title)["page"]
        annotations_coo = coo_dataset.get_annotation(title)["page"]

        df = df_onom_dataset[df_onom_dataset.title==title]
        for _, row in df.iterrows():
            polygon = [tuple(map(int, s.split(","))) for s in row.polygon.split("|")]
            min_x = min(t[0] for t in polygon)
            min_y = min(t[1] for t in polygon)
            max_x = max(t[0] for t in polygon)
            max_y = max(t[1] for t in polygon)
            bb_polygon = BoundingBox(min_x, min_y, max_x, max_y)

            page = annotations[int(row.i_page)]
            panels = page["frame"]
            bb_panels = [BoundingBox(*(panel.get(x) for x in ["@xmin", "@ymin", "@xmax", "@ymax"])) for panel in panels]
            bb_intersection = [(bb_polygon * bb).area for bb in bb_panels]
            i_max = np.argmax(bb_intersection)
            fitting_panel = bb_panels[i_max]

            onom_combined_panel = fitting_panel + bb_polygon

            # Calculate all of the onoms contained in the panel
            page_coo = annotations_coo[int(row.i_page)]
            l_onomatopoeia = page_coo["onomatopoeia"]
            l_contained = []
            bb = shapely.geometry.box(*onom_combined_panel.tuple)
            for d_onom in l_onomatopoeia:
                poly = get_onom_vertices(d_onom)
                poly = shapely.geometry.Polygon(poly)
                if bb.contains(poly):
                    l_contained.append(d_onom["#text"])

            s_contained = ",".join(l_contained)

            r = row.copy()
            r["xmin"] = onom_combined_panel.tuple[0]
            r["ymin"] = onom_combined_panel.tuple[1]
            r["xmax"] = onom_combined_panel.tuple[2]
            r["ymax"] = onom_combined_panel.tuple[3]
            r["contained_onoms"] = s_contained
            l_ret.append(r)

    df_ret = pd.DataFrame(l_ret)
    df_ret = df_ret.sort_values(by="onom_id").reset_index(drop=True)
    df_ret["impath"] = images_outpath + "/raw/onom_panel_" + df_ret.onom_id + ".png"
    df_ret["category"] = "raw"

    #====================================
    # Make the images
    #====================================
    def process_row(item):
        _, row = item

        impath_in = f"{manga109_base_dir}/images/{row.title}/{row.i_page:03d}.jpg"
        image = Image.open(impath_in)

        # Make the undedited panel
        image_panel = image.crop((
            row.xmin,
            row.ymin,
            row.xmax,
            row.ymax,
        ))
        image_panel.save(row.impath)

        # Crop out all of the onomatopoeia
        coo_dataset = manga109api.Parser(root_dir=coo_base_dir)
        annotations = coo_dataset.get_annotation(row.title)["page"]
        page = annotations[row.i_page]

        # Crop out all of the polygons
        for bb_onom in page["onomatopoeia"]:
            onom_vertices = get_onom_vertices(bb_onom)

            image = crop_polygon(image, onom_vertices)

        onom_cropped_impath = images_outpath + "/onom_cropped/onom_panel_" + row.onom_id + ".png"

        image_onom_cropped = image.crop((
            row.xmin,
            row.ymin,
            row.xmax,
            row.ymax,
        ))
        image_onom_cropped.save(onom_cropped_impath)

    print("Saving images...")
    os.makedirs(f"{images_outpath}/raw", exist_ok=True)
    os.makedirs(f"{images_outpath}/onom_cropped", exist_ok=True)

    with Pool(processes=8) as pool:
        pool.map(process_row, tqdm(df_ret.iterrows(), total=df_ret.shape[0]))

    # Make the data for the onom cropped versions
    df_onom_cropped = df_ret.copy()

    df_onom_cropped["impath"] = images_outpath + "/onom_cropped/onom_panel_" + df_onom_cropped.onom_id + ".png"
    df_onom_cropped["category"] = "onom_cropped"

    df_all = pd.concat([df_ret, df_onom_cropped])
    df_all.to_csv(csv_outpath)
