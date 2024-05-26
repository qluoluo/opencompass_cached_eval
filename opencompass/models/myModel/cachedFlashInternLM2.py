import os
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList
import torch.nn.functional as F

# from transformers import LlamaForCausalLM
from transformers import AutoConfig

# from .flash_utils.modeling_internlm2_cached_flash_attn import InternLM2ForCausalLM
# from .flash_utils.AttnCache import AttnCacheConfig
from .flash_utils_v2.modeling_internlm2_cached_flash_attn import InternLM2ForCausalLM
from .flash_utils_v2.AttnCache import AttnCacheConfig

PromptType = Union[PromptList, str]

from ..huggingface import HuggingFaceCausalLM, BaseModel

def print_gpu_memory_info():
    # 获取当前设备的显存信息
    current_device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(current_device).total_memory
    allocated_memory = torch.cuda.memory_allocated(current_device)
    cached_memory = torch.cuda.memory_cached(current_device)
    
    # 转换为GB单位
    total_memory_gb = total_memory / (1024 ** 3)
    allocated_memory_gb = allocated_memory / (1024 ** 3)
    cached_memory_gb = cached_memory / (1024 ** 3)
    
    # 计算使用百分比
    usage_percentage = (allocated_memory / total_memory) * 100
    
    print('*'*100)
    print(f"Device: {current_device}")
    print(f"Total Memory: {total_memory_gb:.2f} GB")
    print(f"Allocated Memory: {allocated_memory_gb:.2f} GB ({usage_percentage:.2f}% used)")
    print(f"Cached Memory: {cached_memory_gb:.2f} GB")
    print('*'*100)

@MODELS.register_module()
class CachedFlashInternLM2CausalLM(HuggingFaceCausalLM):
    def __init__(self,
                 path: str,
                 hf_cache_dir: Optional[str] = None,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 tokenizer_only: bool = False,
                 model_kwargs: dict = dict(device_map='auto'),
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False,
                 pad_token_id: Optional[int] = None,
                 mode: str = 'none',
                 use_fastchat_template: bool = False,
                 end_str: Optional[str] = None,
                 attn_cache_config = None,
                 long_bench_cat = -1,
                 prompt_format: str = '{prompt}',
                 internlm2_attn_implementation: str = 'eager'):
        BaseModel.__init__(self, path=path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=tokenizer_only,
                         meta_template=meta_template)
        if hf_cache_dir is None:
            hf_cache_dir = os.getenv('HF_MODEL_HUB', None)
        self.logger = get_logger()
        self.pad_token_id = pad_token_id
        assert mode in ['none', 'mid']
        self.mode = mode

        self.attn_cache_config = attn_cache_config
        self.internlm2_attn_implementation = internlm2_attn_implementation

        self._load_tokenizer(path=path,
                             tokenizer_path=tokenizer_path,
                             tokenizer_kwargs=tokenizer_kwargs)
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        if not tokenizer_only:
            self._load_model(path=path,
                             model_kwargs=model_kwargs,
                             peft_path=peft_path)
        self.generation_kwargs = generation_kwargs
        self.use_fastchat_template = use_fastchat_template
        self.end_str = end_str

        self.long_bench_cat = long_bench_cat
        self.prompt_format = prompt_format

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):

        self.config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        self.config.attn_implementation = self.internlm2_attn_implementation

        if self.attn_cache_config is not None:
            # assert type(attn_cache_config) == dict, f"attn_cache_config must be a dict, but got {type(attn_cache_config)}"
            self.attn_cache_config = dict(self.attn_cache_config)
            self.config.attn_cache_config = AttnCacheConfig(**self.attn_cache_config)

        
        self._set_model_kwargs_torch_dtype(model_kwargs)
        self.model = InternLM2ForCausalLM.from_pretrained(path, **model_kwargs,
                                                          config=self.config,
                                                          attn_implementation=self.internlm2_attn_implementation)
        # import ipdb
        # ipdb.set_trace()
        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()
        self.model.generation_config.do_sample = False

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]: 
        """Generate results given a list of inputs."""
        self.model.eval()  # Set the model to evaluation mode
        outputs_text = []
        
        for text in inputs:
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids
            # input_ids = torch.tensor(input_ids)

            if self.long_bench_cat > 0:
                if input_ids.shape[-1] > self.long_bench_cat:
                    input_ids = torch.cat([input_ids[:, : self.long_bench_cat // 2], input_ids[:, - self.long_bench_cat // 2:]], dim=-1).to(device=self.model.device)
                else:
                    input_ids = input_ids.to(device=self.model.device)
            
            print(f"\n\ninput_ids.shape: {input_ids.shape}\n")
            if self.attn_cache_config is not None:
                # chunk_length = 128
                # for i in range(0, input_ids.shape[-1]-1, chunk_length):
                #     input_chunk = input_ids[:, i:min(i+chunk_length, input_ids.shape[-1]-1)]
                #     # self.model.generate(input_chunk, max_new_tokens=0, do_sample=False)
                #     self.model.forward(input_chunk)
                chunk_length = 1024
                i = 0

                while i < input_ids.shape[-1] - 1:
                    input_chunk = input_ids[:, i:min(i + chunk_length, input_ids.shape[-1] - 1)]
                    self.model.forward(input_chunk)
                    i += chunk_length
                    # chunk_length = max(1, int(chunk_length * 0.9))  # 确保chunk_length至少为1
                    # chunk_length = chunk_length
                    # print(f"now chunk pos is {i}")
                print_gpu_memory_info()

                generated_ids = input_ids[:, -1:].clone().detach()
                input_ids = input_ids[:, -1:]
                past_key_values = None

                for _ in range(max_out_len):
                    outputs = self.model.forward(input_ids=input_ids, past_key_values=past_key_values)
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                    input_ids = next_token_id
                    generated_ids = torch.cat((generated_ids, next_token_id), dim=-1)
                    if next_token_id == self.tokenizer.eos_token_id or next_token_id == self.tokenizer.pad_token_id:
                        break

                # output_tokens = self.model.generate(input_ids[:, -1:], max_new_tokens=max_out_len, do_sample=False)
                generated_text = self.tokenizer.decode(generated_ids[0][1:], skip_special_tokens=True)
            else:
                output_tokens = self.model.generate(input_ids, max_new_tokens=max_out_len, do_sample=False)
                generated_text = self.tokenizer.decode(output_tokens[0][input_ids.shape[1]:], skip_special_tokens=True)
            outputs_text.append(generated_text)
            
            def clean_cache_all_attentions(model):
                count = 0
                for name, module in model.named_modules():
                    # if isinstance(module, LlamaCacheAttention):
                        if hasattr(module, 'clean_cache'):
                            module.clean_cache()
                            count += 1
                print(f"Cleared cache for {count} attention modules.")

            clean_cache_all_attentions(self.model)
            torch.cuda.empty_cache()

        return outputs_text