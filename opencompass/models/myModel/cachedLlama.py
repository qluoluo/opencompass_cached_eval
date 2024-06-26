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
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM
import sys
sys.path.append('/remote-home/zgliu/wrote_program/kvcache_experiment')
from AttnCache import AttnCacheConfig
from LlamaCacheAttention import LlamaCacheAttention

PromptType = Union[PromptList, str]

from ..huggingface import HuggingFaceCausalLM, BaseModel

class CachedLlamaCausalLM(HuggingFaceCausalLM):
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
                 prompt_format: str = '{prompt}',):
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
        self._load_tokenizer(path=path,
                             tokenizer_path=tokenizer_path,
                             tokenizer_kwargs=tokenizer_kwargs)
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        if not tokenizer_only:
            self._load_model(path=path,
                             model_kwargs=model_kwargs,
                             peft_path=peft_path,
                             attn_cache_config=attn_cache_config)
        self.generation_kwargs = generation_kwargs
        self.use_fastchat_template = use_fastchat_template
        self.end_str = end_str
        self.long_bench_cat = long_bench_cat
        self.prompt_format = prompt_format
    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None,
                    attn_cache_config = None):
        from transformers import AutoModelForCausalLM

        self.config = AutoConfig.from_pretrained(path)

        if attn_cache_config is not None:
            # assert type(attn_cache_config) == dict, f"attn_cache_config must be a dict, but got {type(attn_cache_config)}"
            attn_cache_config = dict(attn_cache_config)
            self.config.use_cached_attention = True
            self.config.attn_cache_config = AttnCacheConfig(**attn_cache_config)
        else:
            self.config.use_cached_attention = False

        
        self._set_model_kwargs_torch_dtype(model_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs, config=self.config)
        # import ipdb
        # ipdb.set_trace()
        if peft_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model,
                                                   peft_path,
                                                   is_trainable=False)
        self.model.eval()
        self.model.generation_config.do_sample = False
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]: 
        """Generate results given a list of inputs."""
        self.model.eval()  # Set the model to evaluation mode
        outputs = []
        for text in inputs:
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids
            input_ids = torch.tensor(input_ids)
            if self.long_bench_cat > 0:
                if input_ids.shape[-1] > self.long_bench_cat:
                    input_ids = torch.cat([input_ids[:, : self.long_bench_cat // 2], input_ids[:, - self.long_bench_cat // 2:]], dim=-1).to(device=self.model.device)
                else:
                    input_ids = input_ids.to(device=self.model.device)
            input_ids = input_ids.to(self.model.device)
            
            if self.config.use_cached_attention:
                chunk_length = 512
                for i in range(0, input_ids.shape[-1]-1, chunk_length):
                    input_chunk = input_ids[:, i:min(i+chunk_length, input_ids.shape[-1]-1)]
                    self.model.generate(input_chunk, max_new_tokens=0, do_sample=False)
                output_tokens = self.model.generate(input_ids[:, -1:], max_new_tokens=max_out_len, do_sample=False)
                generated_text = self.tokenizer.decode(output_tokens[0][1:], skip_special_tokens=True)
            else:
                output_tokens = self.model.generate(input_ids, max_new_tokens=max_out_len, do_sample=False)
                generated_text = self.tokenizer.decode(output_tokens[0][input_ids.shape[1]:], skip_special_tokens=True)
            outputs.append(generated_text)
            
            def clean_cache_all_attentions(model):
                count = 0
                for name, module in model.named_modules():
                    if isinstance(module, LlamaCacheAttention):
                        if hasattr(module, 'clean_cache'):
                            module.clean_cache()
                            count += 1
                print(f"Cleared cache for {count} attention modules.")

            clean_cache_all_attentions(self.model)

        return outputs
    
    def _single_generate(self, inputs: List[str], max_out_len: int, **kwargs) -> List[str]:
        """Support for single prompt inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        if self.long_bench_cat > 0:
            inputs = [self.prompt_format.format(prompt=prompt) for prompt in inputs]
            input_ids = self.tokenizer(inputs, padding=False, truncation=False)['input_ids']
            input_ids = torch.tensor(input_ids)
            if input_ids.shape[-1] > self.long_bench_cat:
                input_ids = torch.cat([input_ids[:, : self.long_bench_cat // 2], input_ids[:, - self.long_bench_cat // 2:]], dim=-1).to(device=self.model.device)
            else:
                input_ids = input_ids.to(device=self.model.device)
        # elif self.pe_config.get('streaming_enable', False) and self.pe_config.get('memory_option', '') in ['', 'sink']:
        #     input_ids = self.tokenizer(inputs, padding=False, truncation=False)['input_ids']
        #     input_ids = torch.tensor(input_ids)
        #     if input_ids.shape[-1] > self.pe_config['start_size'] + self.pe_config['local_size']:
        #         input_ids = torch.cat([input_ids[:, : self.pe_config['start_size']], input_ids[:, - self.pe_config['local_size']:]], dim=-1).to(device=self.model.device)
        #     else:
        #         input_ids = input_ids.to(device=self.model.device)
        else:
            input_ids = self.tokenizer(inputs, padding=False, truncation=True, max_length=self.max_seq_len)['input_ids']
            input_ids = torch.tensor(input_ids).to(device=self.model.device)
        
        # generation_config = self.generation_config
        # generation_config.max_new_tokens = max_out_len
        # self.logger.info('input_ids give')
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=max_out_len, do_sample=False)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, input_ids.shape[1]:]
        # self.logger.info('outputs return')
        decodeds = self.tokenizer.batch_decode(outputs.cpu().tolist(), skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        return decodeds