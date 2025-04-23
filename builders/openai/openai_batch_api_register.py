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
    for task in tqdm(l_tasks):
        basedir = f"batch_tasks/openai/input_split/{task}"
        for filename in tqdm(os.listdir(basedir)):
            filepath = f"{basedir}/{filename}"
            s = filename.split(".jsonl")[0]
            file_id = f"{task}__{s}"
            model.register_batch(task, filepath, file_id)

print(n_files)
