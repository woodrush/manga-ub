import os
import json
import time
import base64
from datetime import datetime

from .baseclass import BenchmarkModel

# Import private configs
import sys
sys.path.append("private")

try:
    import openai
    from private.private_configs import openai_configs
except:
    pass

def new_formatted_message(prompt, encoded_image):
    if encoded_image is None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            }
        ]
    return messages

class GPT4o(BenchmarkModel):
    """
    - Warning: This model reads the `image` variable in `run_q_a_raw` as an image path instead of a PIL image.
    """
    image_by_impath = True
    model_name = "gpt-4o-2024-05-13"
    max_tokens = 300

    def __init__(self, cuda_id, retries=3, failure_wait_time=10):
        self.cuda_id = cuda_id

        api_key = openai_configs["api_key"]
        self.client = openai.OpenAI(api_key=api_key)

        self.failure_wait_time = failure_wait_time
        self.retries = retries

    def run_q_a_raw(self, image, query):
        """
        - Warning: This model reads the `image` variable in `run_q_a_raw` as an image path instead of a PIL image.
        """
        failure_wait_time = self.failure_wait_time
        ret = None
        n_attempt = 1

        failed = False
        while n_attempt <= self.retries:
            start_time = time.time()
            response = None
            try:
                messages = self.generate_chat_api_message(image, query)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0,
                )

                raw_response = response.to_json()
                ret = raw_response

                failure_wait_time = self.failure_wait_time
                n_attempt = 1
            except:
                failure_wait_time *= 2
                print(f"Attempt {n_attempt} failed.", flush=True)
                if response:
                    print(response, flush=True)
                print(f"Will wait for {failure_wait_time} seconds.", flush=True)
                n_attempt += 1
                failed = True

            end_time = time.time()
            execution_time = end_time - start_time
            if failed and execution_time < failure_wait_time:
                padding_time = failure_wait_time - execution_time
                time.sleep(padding_time)

            if ret:
                return ret

        return "[Response generation failed]"

    def run_q_a_raw_multiple(self, impathlist, query):
        raise NotImplementedError

    @staticmethod
    def generate_chat_api_message(image, query):
        if GPT4o.image_is_null(image):
            encoded_image = None
        else:
            with open(image, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        messages = new_formatted_message(query, encoded_image)
        return messages

    @staticmethod
    def generate_batch_api_message(query, image, custom_id):
        messages = GPT4o.generate_chat_api_message(image, query)
        data = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": GPT4o.model_name,
                "messages": messages,
                "max_tokens": GPT4o.max_tokens,
                "temperature": 0,
            }
        }
        return json.dumps(data, ensure_ascii=False, indent=None, separators=(',', ':'))

    @staticmethod
    def post_process_output(s):
        d_response = json.loads(s)
        content = d_response["choices"][0]["message"]["content"]
        return content

    @staticmethod
    def get_total_tokens(raw_response):
        d_response = json.loads(raw_response)
        response_total_tokens = d_response["usage"]["total_tokens"]
        return response_total_tokens

    # Batch API
    def register_batch(self, task, filepath, file_id):
        print(f"Uploading {filepath} to the OpenAI platform using the API key.")
        start_time = datetime.now()
        batch_input_file = self.client.files.create(
            file=open(filepath, "rb"),
            purpose="batch"
        )
        end_time = datetime.now()
        delta_t = end_time - start_time
        print(f"Done. Elapsed time: {delta_t}")

        batch_input_file_id = batch_input_file.id

        print(f"Batch input file ID: {batch_input_file_id}")
        print(f"Will create a batch for this file:")
        endpoint = "/v1/chat/completions"
        completion_window = "24h"
        print(f"{endpoint=}")
        print(f"{completion_window=}")

        print("Creating batch...")
        start_time = datetime.now()
        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata={
                "task": task,
                "file_id": file_id,
            }
        )
        end_time = datetime.now()
        delta_t = end_time - start_time
        print(f"Done. Elapsed time: {delta_t}")

        print("Batch info:")
        print(batch)

        batch_obj_outdir = f"batch_tasks/openai/batch_obj/{task}"
        batch_obj_outpath = f"{batch_obj_outdir}/{file_id}.json"
        os.makedirs(batch_obj_outdir, exist_ok=True)
        print("Writing batch info to {batch_obj_outpath} ...")
        with open(batch_obj_outpath, "wt") as f:
            f.write(batch.to_json())
        print("Done.")

    def retrieve_batch_status(self, batch_id):
        return self.client.batches.retrieve(batch_id)
