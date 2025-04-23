import os
import pandas as pd
import manga109api
from tqdm import tqdm

from configs import manga109_base_dir, panel_id_to_row_csvpath

if __name__ == "__main__":
    os.makedirs("build/", exist_ok=True)

    manga109 = manga109api.Parser(root_dir=manga109_base_dir)

    l_data = []
    for title in tqdm(manga109.books):
        annotations = manga109.get_annotation(book=title)
        for i_page, page in enumerate(annotations["page"]):
            assert page["@index"] == i_page
            for frame in page["frame"]:
                frame["title"] = title
                frame["i_page"] = i_page
                l_data.append(frame)

    df_data = pd.DataFrame(l_data)
    df_data = df_data.rename({
        "@xmin": "xmin",
        "@ymin": "ymin",
        "@xmax": "xmax",
        "@ymax": "ymax",
        "@id":   "panel_id",
    },axis=1).drop(["type"], axis=1)

    df_data.to_csv(panel_id_to_row_csvpath, index=None)
