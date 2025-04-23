import torch

import sys
sys.path.append('external/LLaVA')

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path


def init_llava_model(model_path, model_base, load_8bit, load_4bit, cuda_id, device):
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, load_8bit, load_4bit, device=device
    )
    if cuda_id is not None:
        model = model.cuda(cuda_id).eval()

    return tokenizer, model, image_processor

def prepare_llava_input_ids(query, model, tokenizer, conv_mode, use_image=True):
    if use_image:
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + query
    else:
        inp = query

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    return input_ids

def prepare_llava_image_tensor(image, model, image_processor):
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    return image_tensor

def run_llava_inference(model, tokenizer, input_ids, kargs):
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            **kargs
        )

    outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return outputs
