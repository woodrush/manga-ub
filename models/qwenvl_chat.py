import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from .baseclass import BenchmarkModel

class QwenVLChat(BenchmarkModel):
    """
    - Warning: This model reads the `image` variable in `run_q_a_raw` as an image path instead of a PIL image.
    - The default model.dtype when both bf16 and fp16 are set to False is torch.float16
    """
    image_by_impath = True

    def __init__(self, cuda_id, use_bf16=False, use_fp16=True, do_sample=False):
        self.cuda_id = cuda_id
        self.do_sample = do_sample

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-VL-Chat", trust_remote_code=True, bf16=use_bf16, fp16=use_fp16
        )
        if self.cuda_id is not None:
            self.model = self.model.cuda(self.cuda_id).eval()

    def run_q_a_raw(self, image, query):
        """
        Warning: This model reads the `image` variable as an image path instead of a PIL image.
        """

        if self.image_is_null(image):
            query = self.tokenizer.from_list_format([
                {'text': query},
            ])
        else:
            query = self.tokenizer.from_list_format([
                {'image': image}, # Warning: `image` is a path to the image, instead of a PIL image
                {'text': query},
            ])

        with torch.inference_mode():
            response, history = self.model.chat(
                self.tokenizer,
                query=query,
                history=None,
                do_sample=self.do_sample,
            )

        return response

    @staticmethod
    def post_process_output(s):
        return s

    def run_q_a_raw_multiple(self, impathlist, query):
        raise NotImplementedError

    @staticmethod
    def post_process_output_multiple(s):
        return s

class QwenVLChat_BF16(QwenVLChat):
    def __init__(self, cuda_id):
        super().__init__(cuda_id, use_bf16=True, use_fp16=False)

class QwenVLChat_do_sample(QwenVLChat):
    def __init__(self, cuda_id):
        super().__init__(cuda_id, use_bf16=True, use_fp16=False, do_sample=True)

class QwenVLChat_BF16_do_sample(QwenVLChat):
    def __init__(self, cuda_id):
        super().__init__(cuda_id, use_bf16=True, use_fp16=False, do_sample=True)
