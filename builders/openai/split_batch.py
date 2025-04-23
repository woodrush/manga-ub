import os

from configs import d_csv_paths
from tqdm import tqdm

build_batch_dir = "batch_tasks/openai/input"

build_batch_outdir = "batch_tasks/openai/input_split"

def split_save_jsonl_file(task, max_size=95 * 1024 * 1024):
    input_file = f"{build_batch_dir}/{task}.jsonl"

    file_count = 0
    current_size = 0
    output_file = None
    
    outpath_base = f"{build_batch_outdir}/{task}"
    os.makedirs(outpath_base, exist_ok=True)

    with open(input_file, "r") as infile:
        outpath = f"{outpath_base}/{task}_part{file_count}.jsonl"
        output_file = open(outpath, "w")
        for line in tqdm(infile):
            line_size = len(line.encode("utf-8"))
            if current_size + line_size > max_size:
                output_file.close()
                file_count += 1
                current_size = 0
                outpath = f"{outpath_base}/{task}_part{file_count}.jsonl"
                output_file = open(outpath, "w")
            
            output_file.write(line)
            current_size += line_size

        if output_file:
            output_file.close()

if __name__ == "__main__":
    for task in tqdm(d_csv_paths.keys()):
        print(task)
        split_save_jsonl_file(task=task)
