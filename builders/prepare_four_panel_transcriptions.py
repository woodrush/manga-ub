import pandas as pd
import numpy as np
import sys
import os
from PIL import Image
from tqdm import tqdm
import manga109api

from configs import manga109_base_dir
from utils import BoundingBox


csv_outdir = "build/panel_id_to_speech_text.csv"
splits_csv_path = "build/four_panel_splits.csv"


def d_text_panel_assoc_to_text_pairs(d_text_panel_assoc):
    for k, v in d_text_panel_assoc.items():
        l_text_bb = sorted(v, key=lambda x: x.x2, reverse=True)
        l_text = [x.d["#text"] for x in l_text_bb]
        l_all = []
        for t in l_text:
            t = t.replace("\n", "")
            t = f"「{t}」"
            l_all.append(t)
        s_all = "\n".join(l_all)
        yield (k, s_all)

if __name__ == "__main__":
    df_splits = pd.read_csv(splits_csv_path)
    df_splits["title_page"] = df_splits.title + "_p" + df_splits.i_page.map(str)

    manga109 = manga109api.Parser(root_dir=manga109_base_dir)

    d_title_page = {}
    for title_page in tqdm(df_splits.title_page.unique()):
        df = df_splits[df_splits.title_page==title_page]
        row = df.iloc[0]
        title = row.title
        i_page = row.i_page
        annotations = manga109.get_annotation(book=title)
        page = annotations["page"][i_page]
        l_panels = [BoundingBox.from_dict(d) for d in page["frame"]]
        l_textbb = [BoundingBox.from_dict(d) for d in page["text"]]

        d_text_panel_assoc = {k:[] for k in [x.d["@id"] for x in l_panels]}
        for textbb in l_textbb:
            maxarea = 1
            maxpanel = None
            for panel in l_panels:
                intersection = (textbb * panel).area
                if intersection > maxarea:
                    maxarea = intersection
                    maxpanel = panel
            if maxpanel is not None:
                d_text_panel_assoc[maxpanel.d["@id"]].append(textbb)
        d_title_page[title_page] = d_text_panel_assoc


    l_text_panel_pairs = []
    for k, v in d_title_page.items():
        l_text_panel_pairs += list(d_text_panel_assoc_to_text_pairs(v))

    df_panel_id_text = pd.DataFrame(l_text_panel_pairs, columns=["panel_id", "text"])
    df_panel_id_text.to_csv(csv_outdir, index=None)
