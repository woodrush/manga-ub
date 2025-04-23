# MangaUB: A Manga Understanding Benchmark for Large Multimodal Models
This is the official repository of MangaUB. Paper: [https://doi.org/10.1109/MMUL.2025.3550451](https://doi.org/10.1109/MMUL.2025.3550451)


## Summary
The procedure for running the benchmark can be summarized as:

```sh
git submodule update --init --recursive

# Prepare external/Manga109_released_2023_12_07
# Prepare external/manga_benchmark_dataset
# Prepare external/kangaiset-master

make dataset

CUDA_VISIBLE_DEVICES=0     make MODEL=cogvlm N_PROCESSES=1 run
CUDA_VISIBLE_DEVICES=0     make MODEL=evovlm_jp_v1_7b N_PROCESSES=1 run
CUDA_VISIBLE_DEVICES=0     make MODEL=llava1_5 N_PROCESSES=1 run
CUDA_VISIBLE_DEVICES=0,1,2 make MODEL=llava1_6 N_PROCESSES=1 OPTIONS="--one-model-multi-gpu" run
CUDA_VISIBLE_DEVICES=0     make MODEL=qwenvl_chat N_PROCESSES=1 run

# Run GPT-4o

make eval
less eval_results/table_all.tex  # View benchmark scores
```

Additionally,

```sh
make test    # Task build consistency test
make stats   # Show benchmark task statistics
```

Below, we will describe the details for running the benchmark.


## Step 0: Requirements
### Submodules
After cloning this repo, run the following to obtain external git repository dependecies:

```sh
git submodule update --init --recursive
```

This will clone the following git submodules:
- https://github.com/manga109/public-annotations/: Annotations for MangaUB (maintained in the Manga109 organization repo)
- https://github.com/ku21fan/COO-Comic-Onomatopoeia/: COO Dataset [3]
- https://github.com/haotian-liu/LLaVA/: LLaVA models


### Datasets
There are two external repositories that must be manually placed in this repository:

| Item             | Path                                     | Location                                                 |
|------------------|------------------------------------------|----------------------------------------------------------|
| Manga109 Dataset | `external/Manga109_released_2023_12_07`  | Please manually obtain it from http://www.manga109.org   |
| KangaiSet [4]    | `external/kangaiset-master`              | Ruddy Théodose et al. [4]                                |

After obtaining these from the corresponding locations, place the root directories under the specified `Path`.
Note that you will have to rename the root directory for the MangaUB dataset to `manga_benchmark_dataset`.


### Python Requirements
- pytorch (we use `torch==2.2.1`)
- Modules specified in requirements.txt


## Step 1: Building the Benchmark Tasks
After preparing the dependencies, run the following to build the benchmark tasks:

```sh
make
```

This will run `make dataset`, which will build all of the necessary prompts and images under `./tasks/`.



## Step 2: Running the Benchmark
### Open-Source Models
The benchmark can be run by:

```sh
CUDA_VISIBLE_DEVICES=0 make MODEL=cogvlm N_PROCESSES=1 run
CUDA_VISIBLE_DEVICES=0 make MODEL=evovlm_jp_v1_7b N_PROCESSES=1 run
CUDA_VISIBLE_DEVICES=0 make MODEL=llava1_5 N_PROCESSES=1 run
CUDA_VISIBLE_DEVICES=0,1,2 make MODEL=llava1_6 N_PROCESSES=1 OPTIONS="--one-model-multi-gpu" run
CUDA_VISIBLE_DEVICES=0 make MODEL=qwenvl_chat N_PROCESSES=1 run
```

For the references and details of each of the models used in our paper, please refer to our paper (under preapration).


### GPT-4o
There are two ways to run GPT-4o:

- Using the normal API
- Using the Batch API

The benchmark uses the model `gpt-4o-2024-05-13`.
In the paper, we use the Batch API.


#### Preparation
This is common to both APIs.

For both methods, create the directory `private/` under the repo root, and create `private/private_configs.py` with the following content:

```python
openai_configs = {
    "api_key": "YOUR-API-KEY",
}
```

#### Using the Normal API
The normal API can be run as the same

```sh
make MODEL=gpt-4o N_PROCESSES=1 run
```

When using the normal API, specify `gpt-4o` as the model name when running `make`.


#### Using the Batch API
Using the Batch API requires a lot of interaction with the GPT-4o server.
Run the following to use the Batch API:

```sh
make openai-batch-jsonl                    # [Local] Generate and split batch files for all of the benchmarks
make openai-batch-register                 # [Run]   Upload and run the splitted batch files using the OpenAI API key
# Wait until all batches finish
make openai-batch-fetch                    # [API]   Fetch benchmark responses
make openai-batch-fetch-failed             # [API]   Fetch failed batch information

# If there are failed batches, run the second round:
    make openai-batch-collect-failed       # [API]   Collect failed prompts to be run in the second round
    make openai-register-failed            # [Run]   Registers the failed prompts for the second round
    # Wait until all batches finish
    vim evaluators/openai/fetch_round2.py  #         Here, manually specify the number of failed prompts (the number of batches to retrieve)
    make openai-batch-fetch-round2         # [API]   Fetch responses for round 2, and include them in the results directory
# endif

make openai-parse-batch-result             # [Local] Parse the GPT-4o responses in the results directory
```

The Batch API sometimes returned an error specifying that the image format is malformed, even though it was formed correctly.
In these cases, by re-sending the exact same requests, we were able to get the requests accepted and received the model's responses.
Therefore, in our paper, in the case when the Batch API does not accept our request and we do not get any model responses,
we resend the requests until the API accepts them.

For this purpose, the Batch API is separated into two rounds.
The first round sends all of the requests forming the benchmark.
We then collect all of the rejected batches, and resend only the rejected batches.
In our paper, running the second round was sufficient for obtaining responses for all of the tasks.

Note that when running the recipe `make openai-batch-fetch-round2`, you must manually edit `evaluators/openai/fetch_round2.py` and specify the number of failed batches, i.e., the number batches to fetch from the server.
Running this recipe will fill in the missing responses in the first round, to obtain a complete set of responses for each of the tasks in the benchmark.

The final step `make openai-parse-batch-result` will parse all of the responses to form a CSV file that is compatible for analysis of the rest of the models.


## Step 3: Evaluating the Benchmark Scores
Once all of the model responses are obtained, the benchmark scores can be evaluated with:

```sh
make eval
```

This will generate `eval_results/table_all.tex`, which contains a table of all of the scores for all of the models.


## Miscellaneous
### Task Build Consistency Test
This step is not mandatory for running the benchmark.

The benchmark uses seeded RNGs for building the tasks.
To ensure that the tasks are always built consistently, there is a test that ensures that two different builds of the task exactly match, including all of the images.
It basically runs `make dataset` twice, and runs `diff -r` on both of the results to ensure that the results match exactly.

To run the test, run:

```sh
make test
```

This will build the entire dataset twice, and compare them with `diff -r`.
The comparison run during `make test` is something close to:

```sh
diff -r tmp/tasks_1 tmp/tasks_2
diff -r tmp/build_1 tmp/build_2
```

Here, `./tasks/` contains the actual task definitions, `./build/` is the build directory for building the tasks,
and the directories under `tmp/` are the results for each run in `make test`.


### Show Benchmark Task Statistics
The statistics such as the number of data and prompts can be obtained with:

```sh
make stats
```

This will generate `stats.tex`, which contains a table of the benchmark task statistics.


## Citation
When using MangaUB or if you find our work helpful, please cite our following [paper](https://doi.org/10.1109/MMUL.2025.3550451):
```bibtex
@article{mangaub2025,
  author={Ikuta, Hikaru and Wohler, Leslie and Aizawa, Kiyoharu},
  journal={IEEE MultiMedia},
  title={MangaUB: A Manga Understanding Benchmark for Large Multimodal Models},
  year={2025},
  pages={1-10},
  doi={10.1109/MMUL.2025.3550451}
}
```

Our benchmark is based on the following datasets:
- Manga109 dataset: [1,2]
- COO dataset: [3]
- KangaiSet dataset: [4]


## References
- [1] Y. Matsui, K. Ito, Y. Aramaki, A. Fujimoto, T. Ogawa, T. Yamasaki, and K. Aizawa, “Sketch-based manga retrieval using Manga109 dataset,” <i>Multimedia Tools Appl.,</i> vol. 76, no. 20, pp. 21 811–21 838, 2017, doi: 10.1007/s11042-016-4020-z.
- [2] K. Aizawa, A. Fujimoto, A. Otsubo, T. Ogawa, Y. Matsui, K. Tsubota, and H. Ikuta, “Building a manga dataset “Manga109” with annotations for multimedia applications,” <i>IEEE MultiMedia,</i> vol. 27, no. 2, pp. 8–18, 2020, doi: 10.1109/mmul.2020.2987895.
- [3] J. Baek, Y. Matsui, and K. Aizawa, “COO: Comic onomatopoeia dataset for recognizing arbitrary or truncated texts,” in <i>Proc. European Conf. Comput. Vision,</i> 2022, p. 267–283, doi: 10.1007/978-3-031-19815-1 16.
- [4] R. Théodose and J.-C. Burie, “KangaiSet: A dataset for visual emotion recognition on manga,” in <i>Document Analysis and Recognition: Proc. ICDAR 2023 Workshops,</i> 2023, pp. 120–134, doi: 10.1007/978-3-031-41498-5 9.

For the references and details of each of the models used in our paper, please refer to our [paper](https://arxiv.org/abs/2407.19034).
