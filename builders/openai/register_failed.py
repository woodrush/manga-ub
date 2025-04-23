import os
import sys
sys.path.append(".")

from tqdm import tqdm

from models.gpt4o import GPT4o

l_tasks = [
    "character_count",
    "emotion_benchmark_split",
    "onomatopoeia_scene",
    "recognition_background",
    "panel_localization",
    "next_panel_inference",
]

n_files = 0
if __name__ == "__main__":
    model = GPT4o(cuda_id=0)
    basedir = "batch_tasks/openai/input_split_failed_round2"
    l_failed_tasks = [x for x in os.listdir(basedir) if x in l_tasks]
    for task in tqdm(l_failed_tasks):
        task_dir = f"batch_tasks/openai/input_split_failed_round2/{task}"
        for filename in tqdm(os.listdir(task_dir)):
            filepath = f"{task_dir}/{filename}"
            s = filename.split(".jsonl")[0]
            file_id = f"{task}__{s}_failed_round2"
            model.register_batch(task, filepath, file_id)

print(n_files)
