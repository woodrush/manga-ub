import torch
import re
from transformers import AutoModelForCausalLM, LlamaTokenizer

from .baseclass import BenchmarkModel


class CogVLM(BenchmarkModel):
    def __init__(self, cuda_id, use_fp16=False):
        self.cuda_id = cuda_id

        if use_fp16:
            self.precision = torch.float16
        else:
            self.precision = torch.bfloat16

        self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        self.model = AutoModelForCausalLM.from_pretrained(
            'THUDM/cogvlm-chat-hf',
            torch_dtype=self.precision,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        if self.cuda_id is not None:
            self.model = self.model.cuda(self.cuda_id).eval()

    def run_q_a_raw(self, image, query):
        if self.image_is_null(image):
            inputs = self.model.build_conversation_input_ids(self.tokenizer, query=query, history=[], template_version='vqa')   # vqa mode
            inputs = {
                'input_ids': inputs['input_ids'].unsqueeze(0).cuda(self.cuda_id),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).cuda(self.cuda_id),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).cuda(self.cuda_id),
            }
        else:
            inputs = self.model.build_conversation_input_ids(self.tokenizer, query=query, history=[], images=[image], template_version='vqa')   # vqa mode
            inputs = {
                'input_ids': inputs['input_ids'].unsqueeze(0).cuda(self.cuda_id),
                'token_type_ids': inputs['token_type_ids'].unsqueeze(0).cuda(self.cuda_id),
                'attention_mask': inputs['attention_mask'].unsqueeze(0).cuda(self.cuda_id),
                'images': [[inputs['images'][0].cuda(self.cuda_id).to(self.precision)]],
            }

        gen_kwargs = {
            "max_length": 2048,
            "do_sample": False,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]

        return str(self.tokenizer.decode(outputs[0]))

    def run_q_a_raw_multiple(self, impathlist, query):
        raise NotImplementedError

    @staticmethod
    def post_process_output(s):
        s = re.sub(r'</s>$', '', s)
        return s

class CogVLM_FP16(CogVLM):
    def __init__(self, cuda_id):
        super().__init__(cuda_id, use_fp16=True)
