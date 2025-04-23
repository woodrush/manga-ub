import torch
import re
from transformers import AutoModelForVision2Seq, AutoProcessor

from .baseclass import BenchmarkModel


class EvoVLM_JP_v1_7B(BenchmarkModel):
    def __init__(self, cuda_id):
        self.cuda_id = cuda_id
        model_name = "SakanaAI/EvoVLM-JP-v1-7B"

        self.max_new_tokens = 64

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        if self.cuda_id is not None:
            self.model = self.model.cuda(self.cuda_id).eval()

    def run_q_a_raw(self, image, query):
        if self.image_is_null(image):
            messages = [
                {"role": "user", "content": f"{query}"},
            ]
            input_ids = self.processor.tokenizer.apply_chat_template(messages, return_tensors="pt").cuda(self.cuda_id)

            inputs = {
                "input_ids": input_ids,
            }
        else:
            messages = [
                {"role": "user", "content": f"<image>\n{query}"},
            ]
            input_ids = self.processor.tokenizer.apply_chat_template(messages, return_tensors="pt").cuda(self.cuda_id)

            pixel_values = self.processor.image_processor(images=image, return_tensors="pt")["pixel_values"].cuda(self.cuda_id)
            inputs = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
            }
        gen_kwargs = {
            "do_sample": False,
            "max_new_tokens": self.max_new_tokens,
        }

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        
        response_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(response_ids, skip_special_tokens=True)[0]
        
        return response

    def run_q_a_raw_multiple(self, impathlist, query):
        raise NotImplementedError

    @staticmethod
    def post_process_output(s):
        return s.strip()
