import torch
import re

from .baseclass import BenchmarkModel

from external.llava_utils.llava_utils import init_llava_model, prepare_llava_input_ids, prepare_llava_image_tensor, run_llava_inference


class LLaVA_Base(BenchmarkModel):
    """
    - Default model.torch.dtype: torch.float16
    """
    def __init__(self, cuda_id, conv_mode, n_bits=None, model_path=None, model_base=None):
        self.cuda_id = cuda_id
        if cuda_id is not None:
            self.device = f"cuda:{self.cuda_id}"
        else:
            self.device = "cuda"

        self.conv_mode = conv_mode
        self.max_new_tokens = 512

        load_8bit = n_bits == 8
        load_4bit = n_bits == 4

        self.tokenizer, self.model, self.image_processor = init_llava_model(
            model_path, model_base, load_8bit, load_4bit, self.cuda_id, self.device
        )

    def run_q_a_raw(self, image, query):
        if self.image_is_null(image):
            input_ids = prepare_llava_input_ids(query, self.model, self.tokenizer, self.conv_mode, use_image=False)
            kargs = {
                "do_sample": False,
                "max_new_tokens": self.max_new_tokens,
                "use_cache": True,
            }
        else:
            input_ids = prepare_llava_input_ids(query, self.model, self.tokenizer, self.conv_mode, use_image=True)
            image_tensor = prepare_llava_image_tensor(image, self.model, self.image_processor)
            kargs = {
                "images": image_tensor,
                "image_sizes": [image.size],
                "do_sample": False,
                "max_new_tokens": self.max_new_tokens,
                "use_cache": True,
            }

        outputs = run_llava_inference(
            self.model, self.tokenizer, input_ids, kargs
        )

        return outputs

    @staticmethod
    def post_process_output(s):
        return s

    def run_q_a_raw_multiple(self, impathlist, query):
        raise NotImplementedError

class LLaVA1_5(LLaVA_Base):
    conv_mode = "llava_v1"

    def __init__(self, cuda_id, n_bits=None, model_path=None, model_base=None):
        if model_path is None:
            model_path = "liuhaotian/llava-v1.5-13b"
        super().__init__(cuda_id, self.conv_mode, n_bits=n_bits, model_path=model_path, model_base=model_base)

class LLaVA1_6(LLaVA_Base):
    conv_mode = "chatml_direct"

    def __init__(self, cuda_id, n_bits=None):
        model_path = "liuhaotian/llava-v1.6-34b"
        super().__init__(cuda_id, self.conv_mode, n_bits=n_bits, model_path=model_path, model_base=None)

class LLaVA1_5_4bit(LLaVA1_5):
    def __init__(self):
        super().__init__(n_bits=4)

class LLaVA1_5_8bit(LLaVA1_5):
    def __init__(self):
        super().__init__(n_bits=8)

class LLaVA1_6_4bit(LLaVA1_6):
    def __init__(self):
        super().__init__(n_bits=4)

class LLaVA1_6_8bit(LLaVA1_6):
    def __init__(self):
        super().__init__(n_bits=8)
