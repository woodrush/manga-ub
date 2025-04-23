from .cogvlm import CogVLM, CogVLM_FP16
from .evovlm_jp import EvoVLM_JP_v1_7B
from .gpt4o import GPT4o
from .llava_models import LLaVA1_5, LLaVA1_5_4bit, LLaVA1_5_8bit, LLaVA1_6, LLaVA1_6_4bit, LLaVA1_6_8bit
from .qwenvl_chat import QwenVLChat, QwenVLChat_BF16, QwenVLChat_do_sample, QwenVLChat_BF16_do_sample

d_models = {
    "cogvlm": CogVLM,
    "cogvlm_fp16": CogVLM_FP16,
    "evovlm_jp_v1_7b": EvoVLM_JP_v1_7B,
    "gpt4o": GPT4o,
    "llava1_5": LLaVA1_5,
    "llava1_5_4bit": LLaVA1_5_4bit,
    "llava1_5_8bit": LLaVA1_5_8bit,
    "llava1_6": LLaVA1_6,
    "llava1_6_4bit": LLaVA1_6_4bit,
    "llava1_6_8bit": LLaVA1_6_8bit,
    "qwenvl_chat": QwenVLChat,
    "qwenvl_chat_do_sample": QwenVLChat_do_sample,
    "qwenvl_chat_bf16": QwenVLChat_BF16,
    "qwenvl_chat_bf16_do_sample": QwenVLChat_BF16_do_sample,
}

available_models = d_models.keys()

def get_model(s):
    return d_models[s]
