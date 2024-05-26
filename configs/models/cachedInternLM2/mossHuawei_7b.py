# from opencompass.models import HuggingFaceCausalLM
from opencompass.models.myModel.cachedFlashInternLM2 import CachedFlashInternLM2CausalLM
# from ....opencompass.models.myModel.cachedFlashLlama import CachedFlashLlamaCausalLM
from transformers import LlamaConfig
import torch


internlm2_model_path = 'models/mossHuawei'

attn_cache_config = None
internlm2_attn_implementation = 'eager'

USE_CACHED_ATTENTION = True
if USE_CACHED_ATTENTION:
   internlm2_attn_implementation = "cached_flash_attention_2"
   attn_cache_config = {
        "start_size": 4,
        "recent_size": 2048,
        "mid_size": 512,

        "key_compress_method": 'cut-head-suffix',
        "key_reserved_dim": 256,
        "key_compress_split_head": False,
        
        "value_compress_method": "none",
        "value_reserved_dim": 4096,
        "value_compress_split_head": False,

        "new_decompress_method": True,
        "max_storage_mid_size": -1,
   }

models = [
    dict(
        type=CachedFlashInternLM2CausalLM,
        abbr=f'cachedinternlm2-7b-{attn_cache_config["key_compress_method"]}-{attn_cache_config["key_reserved_dim"]}-{attn_cache_config["value_compress_method"]}-{attn_cache_config["value_reserved_dim"]}',
        path=internlm2_model_path,
        tokenizer_path=internlm2_model_path,
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              trust_remote_code=True,
                              ),
        max_out_len=500,
        # max_seq_len=4096,
        batch_size=1,
        model_kwargs=dict(device_map='cuda',
                          torch_dtype=torch.bfloat16,
                          trust_remote_code=True,
                          ),
        internlm2_attn_implementation=internlm2_attn_implementation,
        attn_cache_config=attn_cache_config,
        long_bench_cat=31500,
        # long_bench_cat=300,

        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]