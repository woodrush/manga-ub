import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import time

import sys
sys.path.append(".")
from models.gpt4o import GPT4o


if __name__ == "__main__":
    model = GPT4o(cuda_id=0)

    print("get_batch_output_file_list ...")

    all_batches = []

    pbar = tqdm(total=None)
    response = model.client.batches.list(after=None)

    # There are a total of 138 batches
    for _ in range(1):
        pbar.update(1)
        time.sleep(0.5)
        all_batches.extend(response.data)

        if not response.has_next_page():
            break

        response = response.get_next_page()

    pbar.close()

    assert len(all_batches) == 20

    all_batches = all_batches[:4]

    print("Writing output to file...")
    os.makedirs("batch_tasks_output/openai", exist_ok=True)

    for batch in tqdm(all_batches):
        if batch.output_file_id is None:
            print("Skipping", batch)
            continue
        content = model.client.files.content(batch.output_file_id)
        time.sleep(0.5)
        task = batch.metadata["task"]
        file_id = batch.metadata["file_id"]
        out_basepath = f"batch_tasks_output/openai/{task}"
        out_filename = f"{file_id}.jsonl"
        os.makedirs(out_basepath, exist_ok=True)
        out_filepath = f"{out_basepath}/{out_filename}"
        content.write_to_file(out_filepath)
