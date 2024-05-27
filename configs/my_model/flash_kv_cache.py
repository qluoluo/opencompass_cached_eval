
from opencompass.models.flash_kv_cache_wrapper import FlashKVModel

import numpy as np

paths = {'llama2_7B': '/fs-computility/llm/shared/llm_data/llm_llama/llama2/llama-2-7b-hf/', 
         'llama2_13B': '/fs-computility/llm/shared/llm_data/llm_llama/llama2/llama-2-13b-hf/', 
         'internlm2_7B': '/fs-computility/llm/shared/zhangshuo/exported_transformers/official_Ampere_7B_Enhance_1.2.0rc/50000/',
         'huawei_moss': '/fs-computility/llm/shared/liuxiaoran/tmp_ckpts_hf/long_score-internlm2_7B-b1000000_0127/2000/',  
         }

num_gpus = {'llama2_7B': 1, 'llama2_13B': 2, 'internlm2_7B': 1, 'huawei_moss': 1, }

# cache_config = {'recent_length', 'global_length', 'middle_length', 
#                 'prefill_strategy',  # ['full-attn', 'streaming', ]
#                 'generate_strategy',  # ['full-attn', 'streaming', ]
#                 'compress_strategy',  # ['none', 'cut-random', 'cut-prefix', 'cut-suffix', 'cut-head-prefix', 'cut-head-suffix', ]
#                 'reserved_dim', 'streaming_stride', }

tags = [
        ('-ntk-32k_cat', 'llama2_7B', 'llama2_7B', 'llama', '{prompt}', 31500, 
         {'recent_length': 2048, 'global_length': 4, 'middle_length': 1024, 'compress_strategy': 'cut-head-suffix', 'reserved_dim': 1024, }), 
        ]  # ['none', 'cut-random', 'cut-prefix', 'cut-suffix', 'cut-head-prefix', , ]

models = [
    dict(
        abbr='{}{}'.format(group, abbr),  # name in path
        type=FlashKVModel, 
        cache_config=cache_config,
        model_type=model_type, 
        model_path=paths[path],
        config_path=paths[group],
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left',
                              trust_remote_code=True, use_fast=False, ),
        max_out_len=128,  # no use
        max_seq_len=16384,
        long_bench_cat=cat_len, 
        prompt_format=prompt_format, 
        batch_size=1, 
        batch_padding=True,
        run_cfg=dict(num_gpus=num_gpus[group.split('-')[0]], 
                     num_procs=num_gpus[group.split('-')[0]]),  # tp or pp size
    ) for abbr, group, path, model_type, prompt_format, cat_len, cache_config in tags]

