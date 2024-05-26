# from opencompass.models import HuggingFaceCausalLM
from opencompass.models.myModel.cachedFlashLlama import CachedFlashLlamaCausalLM
# from ....opencompass.models.myModel.cachedFlashLlama import CachedFlashLlamaCausalLM
from transformers import LlamaConfig
import torch


llama_model_path = "/remote-home/share/models/llama_v2_hf/7b"

attn_cache_config = None
llama_attn_implementation = 'eager'

USE_CACHED_ATTENTION = True
if USE_CACHED_ATTENTION:
   llama_attn_implementation = "cached_flash_attention_2"
   attn_cache_config = {
        "start_size": 4,
        "recent_size": 2048,
        "mid_size": 512,

        "key_compress_method": 'cut-head-suffix',
        "key_reserved_dim": 1024,
        "key_compress_split_head": False,
        
        "value_compress_method": "none",
        "value_reserved_dim": 4096,
        "value_compress_split_head": False,

        "new_decompress_method": True,
        "max_storage_mid_size": -1,
   }

models = [
    # LLaMA 7B
    dict(
        type=CachedFlashLlamaCausalLM,
        abbr=f'cachedllama2-7b-{attn_cache_config["key_compress_method"]}-{attn_cache_config["key_reserved_dim"]}-{attn_cache_config["value_compress_method"]}-{attn_cache_config["value_reserved_dim"]}',
        path=llama_model_path,
        tokenizer_path=llama_model_path,
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=500,
        # max_seq_len=4096,
        batch_size=1,
        model_kwargs=dict(device_map='cuda',
                          torch_dtype=torch.float16,
                          ),
        llama_attn_implementation=llama_attn_implementation,
        attn_cache_config=attn_cache_config,
        long_bench_cat=31500,
        # long_bench_cat=300,

        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]