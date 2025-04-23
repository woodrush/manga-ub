all: dataset
test: test-dataset-consistency
run: run-all-tasks
eval: eval-results-all
stats: stats.tex

#====================================================================================
# Dataset preparation
#====================================================================================
MANGA109_DATASET := external/Manga109_released_2023_12_07
MANGABENCH_ANNOTATIONS := external/public-annotations/MangaUB-Annotations
KANGAISET_BASE := external/kangaiset-master
COO_DATASET := external/COO-Comic-Onomatopoeia/COO-data/books.txt
KANGAISET := $(KANGAISET_BASE)/images

EXTERNAL_DEPS := \
	$(MANGA109_DATASET) \
	$(KANGAISET) \
	$(MANGABENCH_ANNOTATIONS) \
	$(COO_DATASET)

dataset: $(EXTERNAL_DEPS) csvs

csvs:		build/panel_id_to_row.csv \
			build/prepare_panel_localization.csv \
			build/four_panel_splits.csv \
			build/panel_id_to_speech_text.csv \
			build/next_panel_inference/combinations/test.csv \
			build/onomatopoeia_scene_panels.csv \
			tasks/recognition_background.csv \
			tasks/character_count.csv \
			tasks/emotion_benchmark_split.csv \
			tasks/onomatopoeia_scene.csv \
			tasks/panel_localization.csv \
			tasks/next_panel_inference/test.csv

# Preparation
build/panel_id_to_row.csv: \
			builders/prepare_panel_id_dict.py \
			$(MANGA109_DATASET)

	python builders/prepare_panel_id_dict.py


# Background recognition tasks
tasks/recognition_background.csv: \
			builders/recognition_background.py \
			builders/recognition_background_images.py \
			build/panel_id_to_row.csv \
			$(MANGABENCH_ANNOTATIONS) \
			$(MANGA109_DATASET)

	python builders/recognition_background.py
	python builders/recognition_background_images.py

# Character count task
tasks/character_count.csv: \
			builders/character_count.py \
			builders/character_count_images.py \
			$(MANGABENCH_ANNOTATIONS) \
			$(MANGA109_DATASET)

	python builders/character_count.py
	python builders/character_count_images.py

# Panel localization task
build/prepare_panel_localization.csv: \
			builders/prepare_panel_localization.py \
			build/panel_id_to_row.csv \
			$(MANGABENCH_ANNOTATIONS)

	python builders/prepare_panel_localization.py

tasks/panel_localization.csv: \
			builders/panel_localization.py \
			builders/panel_localization_images.py \
			build/prepare_panel_localization.csv

	python builders/panel_localization.py
	python builders/panel_localization_images.py

# Next panel inference
build/four_panel_splits.csv: \
			builders/split_four_panel_comics.py \
			build/panel_id_to_row.csv \
			$(MANGABENCH_ANNOTATIONS)

	python builders/split_four_panel_comics.py

build/panel_id_to_speech_text.csv: \
			builders/prepare_four_panel_transcriptions.py \
			build/four_panel_splits.csv

	python builders/prepare_four_panel_transcriptions.py

build/next_panel_inference/combinations/test.csv: \
			builders/prepare_next_panel_inference.py \
			build/panel_id_to_row.csv \
			build/four_panel_splits.csv \
			build/panel_id_to_speech_text.csv \
			$(MANGABENCH_ANNOTATIONS)

	python builders/prepare_next_panel_inference.py

tasks/images/next_panel_inference/labels/test.csv: \
			builders/prepare_four_panel_images.py \
			builders/next_panel_inference_images.py \
			build/next_panel_inference/combinations/test.csv \
			build/four_panel_splits.csv

	python builders/prepare_four_panel_images.py
	python builders/next_panel_inference_images.py

tasks/next_panel_inference/test.csv: \
			builders/next_panel_inference.py \
			build/next_panel_inference/combinations/test.csv \
			tasks/images/next_panel_inference/labels/test.csv \
			$(MANGA109_DATASET)

	python builders/next_panel_inference.py

# Onomatopoeia scene
build/onomatopoeia_scene_panels.csv: \
			builders/prepare_onomatopoeia_scene.py \
			$(COO_DATASET) \
			$(MANGA109_DATASET)

	python builders/prepare_onomatopoeia_scene.py

tasks/onomatopoeia_scene.csv: \
			builders/onomatopoeia_scene.py \
			build/onomatopoeia_scene_panels.csv \
			$(MANGABENCH_ANNOTATIONS) \
			$(MANGABENCH_ANNOTATIONS)/annotations/onomatopoeia_descriptions.json

	python builders/onomatopoeia_scene.py

# Emotion task
tasks/emotion_benchmark_split.csv: \
			builders/emotion_benchmark_split.py \
			$(KANGAISET)

	python builders/emotion_benchmark_split.py

# External datasets
$(MANGA109_DATASET):
	$(error Please manually place the Manga109 dataset's root directory as `$(MANGA109_DATASET)`. Please see README.md for more details.)

$(KANGAISET):
	$(error Please manually place the KangaiSet dataset repository as `$(KANGAISET_BASE)` and run the dataset preparation script inside it and generate `$(KANGAISET)`. Please see README.md for more details.)

$(MANGABENCH_ANNOTATIONS):
	$(error Please pull the public-annotations git submodule with `git submodule update --init --recursive`. Please see README.md for more details.)

$(COO_DATASET):
	$(error Please pull the COO-Comic-Onomatopoeia dataset git submodule with `git submodule update --init --recursive`. Please see README.md for more details.)

external/LLaVA/llava:
	$(error Please pull the LLaVA git submodule with `git submodule update --init --recursive`. Please see README.md for more details.)

private/private_configs.py:
	$(error Please place `private/private_configs.py` holding the API key information. Please see README.md for more details.)


#====================================================================================
# Dataset stats
#====================================================================================
stats.tex: tools/collect_stats.py
	python tools/collect_stats.py > stats.tex

visualize: tools/visualize_data.py
	python tools/visualize_data.py

#====================================================================================
# Task build consistency test
#====================================================================================
TIMESTAMP := $(shell date +%Y%m%d-%H%M%S)
TMP_DIR := tmp/test-$(TIMESTAMP)
FIRST_TASKS := $(TMP_DIR)/run_1
SECOND_TASKS := $(TMP_DIR)/run_2

# Build the dataset twice and check that it generates identical files
test-dataset-consistency:
	mkdir -p $(TMP_DIR)
	mkdir -p $(FIRST_TASKS)
	mkdir -p $(SECOND_TASKS)

	@if [ -d "tasks" ]; then mv tasks tasks.bak; fi
	@if [ -d "build" ]; then mv build build.bak; fi

	$(MAKE) dataset
	mv tasks $(FIRST_TASKS)
	mv build $(FIRST_TASKS)

	$(MAKE) dataset
	mv tasks $(SECOND_TASKS)
	mv build $(SECOND_TASKS)

	@if [ -d "tasks.bak" ]; then mv tasks.bak tasks; fi
	@if [ -d "build.bak" ]; then mv build.bak build; fi

	diff -r $(FIRST_TASKS) $(SECOND_TASKS)


#====================================================================================
# Task runners
#====================================================================================
N_PROCESSES ?= 1
MODEL ?= None
OPTIONS ?=

run-all-tasks: \
	external/LLaVA/llava \
	run-background-recognition \
	run-character-count \
	run-panel-localization \
	run-onomatopoeia-scene \
	run-next-panel-inference \
	run-emotion-benchmark-split

run-background-recognition:
	python run_tasks.py \
		--model $(MODEL) \
		--n-processes $(N_PROCESSES) \
		$(OPTIONS) \
		--bench-csv "tasks/recognition_background.csv" \
		--task-name recognition_background

run-character-count:
	python run_tasks.py \
		--model $(MODEL) \
		--n-processes $(N_PROCESSES) \
		$(OPTIONS) \
		--bench-csv "tasks/character_count.csv" \
		--task-name character_count

run-panel-localization:
	python run_tasks.py \
		--model $(MODEL) \
		--n-processes $(N_PROCESSES) \
		$(OPTIONS) \
		--bench-csv "tasks/panel_localization.csv" \
		--task-name panel_localization

run-next-panel-inference:
	python run_tasks.py \
		--model $(MODEL) \
		--n-processes $(N_PROCESSES) \
		$(OPTIONS) \
		--bench-csv "tasks/next_panel_inference/test.csv" \
		--task-name next_panel_inference

run-next-panel-inference-trainsplit:
	python run_tasks.py \
		--model $(MODEL) \
		--n-processes $(N_PROCESSES) \
		$(OPTIONS) \
		--bench-csv "tasks/next_panel_inference/train.csv" \
		--task-name next_panel_inference_trainsplit

run-next-panel-inference-validsplit:
	python run_tasks.py \
		--model $(MODEL) \
		--n-processes $(N_PROCESSES) \
		$(OPTIONS) \
		--bench-csv "tasks/next_panel_inference/valid.csv" \
		--task-name next_panel_inference_validsplit

run-onomatopoeia-scene:
	python run_tasks.py \
		--model $(MODEL) \
		--n-processes $(N_PROCESSES) \
		$(OPTIONS) \
		--bench-csv "tasks/onomatopoeia_scene.csv" \
		--task-name onomatopoeia_scene

run-emotion-benchmark-split:
	python run_tasks.py \
		--model $(MODEL) \
		--n-processes $(N_PROCESSES) \
		$(OPTIONS) \
		--bench-csv "tasks/emotion_benchmark_split.csv" \
		--task-name emotion_benchmark_split


#====================================================================================
# OpenAI Batch API input
#====================================================================================
openai-batch-jsonl: \
			builders/openai/openai_batch_api_preparation.py \
			builders/openai/split_batch.py \
			tasks/character_count.csv \
			tasks/emotion_benchmark_split.csv \
			tasks/onomatopoeia_scene.csv \
			tasks/panel_localization.csv \
			tasks/recognition_background.csv \
			tasks/next_panel_inference/test.csv

	python builders/openai/openai_batch_api_preparation.py
	python builders/openai/split_batch.py

openai-batch-register: \
			builders/openai/openai_batch_api_register.py \
			private/private_configs.py

	python builders/openai/openai_batch_api_register.py


#====================================================================================
# OpenAI Batch API results
#====================================================================================
openai-batch-fetch: \
			private/private_configs.py \
			evaluators/openai/fetch_batches.py

	python evaluators/openai/fetch_batches.py

openai-batch-fetch-failed: \
			private/private_configs.py \
			evaluators/openai/fetch_failed.py

	python evaluators/openai/fetch_failed.py

openai-batch-collect-failed: \
			evaluators/openai/collect_failed_prompts.py

	python evaluators/openai/collect_failed_prompts.py

openai-register-failed:	\
			private/private_configs.py \
			builders/openai/register_failed.py

	python builders/openai/register_failed.py

openai-batch-fetch-round2: \
			private/private_configs.py \
			evaluators/openai/fetch_round2.py

	python evaluators/openai/fetch_round2.py

openai-parse-batch-results: \
			evaluators/openai/parse_batch_results.py

	python evaluators/openai/parse_batch_results.py

#====================================================================================
# Evaluate results
#====================================================================================
eval-results-all: \
		eval_results/table/recognition_background.tex \
		eval_results/table/character_count_perlabel.tex \
		eval_results/table/character_count.tex \
		eval_results/table/emotion_benchmark_split_percategory.tex \
		eval_results/table/emotion_benchmark_split.tex \
		eval_results/table/onomatopoeia_scene_percategory.tex \
		eval_results/table/onomatopoeia_scene.tex \
		eval_results/table/panel_localization.tex \
		eval_results/table/next_panel_inference.tex

	$(MAKE) eval-concat


eval_results/table/recognition_background.tex: \
			evaluators/eval_background.py evaluators/utils.py

	python evaluators/eval_background.py

eval_results/table/character_count.tex eval_results/table/character_count_perlabel.tex: \
			evaluators/eval_character_count.py evaluators/utils.py

	python evaluators/eval_character_count.py

eval_results/table/onomatopoeia_scene.tex eval_results/table/onomatopoeia_scene_percategory.tex: \
			evaluators/eval_onomatopoeia_scene.py evaluators/utils.py

	python evaluators/eval_onomatopoeia_scene.py

eval_results/table/panel_localization.tex: \
			evaluators/eval_panel_localization.py evaluators/utils.py

	python evaluators/eval_panel_localization.py

eval_results/table/emotion_benchmark_split.tex eval_results/table/emotion_benchmark_split_percategory.tex: \
			evaluators/eval_emotion_benchsplit.py evaluators/utils.py

	python evaluators/eval_emotion_benchsplit.py

eval_results/table/next_panel_inference.tex: \
			evaluators/eval_next_panel_inference.py evaluators/utils.py

	python evaluators/eval_next_panel_inference.py

eval-concat:
	cd eval_results/table; for file in $$(ls -1 | sort); do \
		echo "Filename: $$file"; \
		cat "$$file"; \
		echo "\n"; \
	done | sed s/_/-/g > ../table_all.tex
