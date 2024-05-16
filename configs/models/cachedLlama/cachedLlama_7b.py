# from opencompass.models import HuggingFaceCausalLM
from opencompass.models.myModel.cachedLlama import CachedLlamaCausalLM
from transformers import LlamaConfig
import torch


llama_model_path = "/remote-home/zgliu/wrote_program/kvcache_experiment/models/Llama-2-7b-chat-hf"

attn_cache_config = None
USE_CACHED_ATTENTION = True
if USE_CACHED_ATTENTION:
   attn_cache_config = {
         "start_size": 4,
         "recent_size": 1024,
         "mid_size": 512,
         "compress_method": 'cut-head-suffix',
         "reserved_dim": 1024,
         "new_decompress_method": True,
         "max_storage_mid_size": -1,
   }

models = [
    # LLaMA 7B
    dict(
        type=CachedLlamaCausalLM,
        abbr=f'cachedllama-7b-hf-{attn_cache_config["compress_method"]}-{attn_cache_config["reserved_dim"]}',
        path=llama_model_path,
        tokenizer_path='/remote-home/zgliu/wrote_program/kvcache_experiment/models/Llama-2-7b-chat-hf',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=100,
        # max_seq_len=4096,
        batch_size=1,
        model_kwargs=dict(device_map='cuda',
                          torch_dtype=torch.float16),

        attn_cache_config=attn_cache_config,
        long_bench_cat=31500,

        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]