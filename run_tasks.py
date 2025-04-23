import torch
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Pool

import os
import argparse
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime
from PIL import Image

from models import get_model, available_models
from models.baseclass import BenchmarkModel


bench_result_snapshot_dir = "snapshots"
bench_result_final_dir = "bench_results"

#==================================================================================================
# Parallel functions
#==================================================================================================
# - Parameters declared as `global` become process-specific and are not shared between processes.
#   Here, the variable `model` becomes specific to each process it was called in.
# - Each process carries a reference to a model loaded to a different GPU.

def process_row_image(item):
    i_row, row = item
    if BenchmarkModel.image_is_null(row.impath):
        return None, row
    image = Image.open(row.impath).convert('RGB')
    return image, row

def init_process(d_args):
    global model
    model_name, in_args, cuda_id = d_args["model_name"], d_args["in_args"], d_args["cuda_id"]
    one_model_multi_gpu = d_args["one_model_multi_gpu"]
    if one_model_multi_gpu:
        cuda_id = None
    model = get_model(model_name)(cuda_id=cuda_id, **in_args)

def process_row_model(item):
    global model
    image, row = item

    if model.image_by_impath:
        raw_response, response = model.run_q_a(
            image=row.impath,
            query=row.prompt,
        )
    else:
        raw_response, response = model.run_q_a(
            image=image,
            query=row.prompt,
        )

    result = {
        "row": row,
        "raw_response": raw_response,
        "response": response,
        "cuda_id": model.cuda_id,
    }

    return result

#==================================================================================================
# Helper functions
#==================================================================================================
def load_prompts(bench_csv, shuffle_task_queue=False):
    df_prompts = pd.read_csv(bench_csv)

    if shuffle_task_queue:
        print(f"Evaluation order randomization is on. Seed: 0")
        rng = np.random.default_rng(seed=0)
        df_prompts_old_shape = df_prompts.shape
        df_prompts = df_prompts.sample(frac=1, random_state=rng).reset_index(drop=True)
        assert df_prompts.shape == df_prompts_old_shape

    return df_prompts

def save_results(
    l_results,
    save_dir,
    task_name,
    timestamp,
    bench_results_suffix,
    model_name
):
    if bench_results_suffix != "":
        bench_results_suffix = f"-{bench_results_suffix}"

    csv_outdir = f"{save_dir}/{task_name}"
    csv_out_filename = f"{model_name}-{timestamp}{bench_results_suffix}.csv"
    csv_outpath = f"{csv_outdir}/{csv_out_filename}"
    
    os.makedirs(csv_outdir, exist_ok=True)

    l_rows = [
        (result["row"], result["raw_response"], result["response"])
        for result in l_results
    ]
    l_rows, l_raw_responses, l_responses = zip(*l_rows)
    df_out = pd.concat([row.to_frame().T for row in l_rows], ignore_index=True)
    df_out["raw_response"] = l_raw_responses
    df_out["response"] = l_responses

    df_out.to_csv(csv_outpath, index=False, header=True)

#==================================================================================================
# Main
#==================================================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help=f"Choose from: {available_models}")
    parser.add_argument("--bench-csv", type=str, required=True)
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--bench-results-suffix", type=str, default="")
    parser.add_argument("--n-processes", type=int, default=1)
    parser.add_argument("--image-loader-poolsize", type=int, default=2)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--result-snapshot-interval", type=int, default=20)
    parser.add_argument("--one-model-multi-gpu", action="store_true")
    parser.add_argument("--shuffle-task-queue", action="store_true")
    parser.add_argument("--monitor-total-token-count", action="store_true")
    args = parser.parse_args()

    # Prepare datetime suffix for saving the results
    now = datetime.now()
    result_filename_timestamp = now.strftime("%Y-%m-%d-%H-%M-%S-%f")

    os.makedirs(bench_result_snapshot_dir, exist_ok=True)
    os.makedirs(bench_result_final_dir, exist_ok=True)

    # Required for LLaVA
    if multiprocessing.get_start_method() == "fork":
        multiprocessing.set_start_method("spawn", force=True)

    assert args.model in available_models, args.model

    if args.monitor_total_token_count:
        assert "get_total_tokens" in dir(get_model(args.model))

    with Pool(processes=args.n_processes) as pool:
        #================================================================
        # Initialize the models in each GPU, in each process
        #================================================================
        in_args = {}
        if args.model_base is not None:
            in_args["model_base"] = args.model_base
        if args.model_path is not None:
            in_args["model_path"] = args.model_path

        initargs = [
            {
                "model_name": args.model,
                "in_args": in_args,
                "cuda_id": cuda_id,
                "one_model_multi_gpu": args.one_model_multi_gpu,
            }
            for cuda_id in range(args.n_processes)
        ]

        # Apply the model initialization function to each process in the pool
        pool.map(init_process, initargs)

        #================================================================
        # Run the prompts in the specified tasks in parallel
        #================================================================
        l_results = []
        n_current_task_cumulative_token_count = 0

        df_prompts = load_prompts(args.bench_csv, args.shuffle_task_queue)
        assert "impath" in df_prompts.columns
        assert "prompt" in df_prompts.columns

        # Use a separate process pool to load image files, for pipelining
        with Pool(processes=args.image_loader_poolsize) as pool_image_loaders:
            # Assign jobs to each process.
            # Each result is immediately brought to the iterator once they have been calculated,
            # so the results are not necessarily in the original order.

            # The pipeline is as follows:
            # df_prompts -> process_row_image -> [(image, row)] -> process_row_model -> [{"row": row, "response": response, ...}]
            process_row_image_iter = pool_image_loaders.imap(process_row_image, df_prompts.iterrows())
            process_row_model_iter = pool.imap(process_row_model, process_row_image_iter)
            for i_iter, result in tqdm(enumerate(process_row_model_iter), total=df_prompts.shape[0],):
                row = result["row"]
                response = result["response"]
                raw_response = result["raw_response"]
                cuda_id = result["cuda_id"]

                print(row)
                print(f"{args.model=}")
                print(f"{args.bench_csv=}")
                print(f"{args.bench_results_suffix=}")
                print(f"{cuda_id=}")
                print(f"{raw_response=}")
                print(f"{response=}", flush=True)

                l_results.append(result)

                if i_iter % args.result_snapshot_interval == 0:
                    save_results(
                        l_results=l_results,
                        save_dir=bench_result_snapshot_dir,
                        task_name=args.task_name,
                        timestamp=result_filename_timestamp,
                        bench_results_suffix=args.bench_results_suffix,
                        model_name=args.model
                    )

                if args.monitor_total_token_count:
                    n_current_task_cumulative_token_count += get_model(args.model).get_total_tokens(raw_response)
                    tokens_per_iter = n_current_task_cumulative_token_count / (i_iter + 1)
                    total_tokens = n_current_task_cumulative_token_count
                    print(f"Cumulative token count for this task: {n_current_task_cumulative_token_count} ({tokens_per_iter} tokens/iter)", flush=True)
                    print(f"Cumulative token count for all tasks: {total_tokens}", flush=True)

            save_results(
                l_results=l_results,
                save_dir=bench_result_snapshot_dir,
                task_name=args.task_name,
                timestamp=result_filename_timestamp,
                bench_results_suffix=args.bench_results_suffix,
                model_name=args.model
            )

            save_results(
                l_results=l_results,
                save_dir=bench_result_final_dir,
                task_name=args.task_name,
                timestamp=result_filename_timestamp,
                bench_results_suffix=args.bench_results_suffix,
                model_name=args.model
            )
