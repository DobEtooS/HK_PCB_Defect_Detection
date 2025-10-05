#引入大模型依赖
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image
import numpy as np
import os
import time
# import spaces  # Import spaces for ZeroGPU compatibility


def load():
    # 引入大模型和语言处理器
    model_path = r".\Janus-Pro-1B"  # 更改为了相对目录，后期可以手动指定/搜素模型地址
    config = AutoConfig.from_pretrained(model_path)
    language_config = config.language_config
    language_config._attn_implementation = 'eager'
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                                  language_config=language_config,
                                                  trust_remote_code=True)

    # 判断电脑GPU情况，有无cuda，没有就移交给cpu处理
    if torch.cuda.is_available():
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
    else:
        vl_gpt = vl_gpt.to(torch.float16)

    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'


#图片文字提示词
@torch.inference_mode()
def multimodal_understanding(self,image_path:str, question:str):
    image = image_path
    question = question
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    pil_images = load_pil_images(conversation)
    prepare_inputs = self.vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(self.vl_gpt.device)
    inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    inputs_embeds
    outputs = self.vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=self.tokenizer.eos_token_id,
        bos_token_id=self.tokenizer.bos_token_id,
        eos_token_id=self.tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )
    answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer
