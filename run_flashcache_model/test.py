import torch
import transformers, datasets
from transformers import AutoTokenizer, AutoConfig
from opencompass.models.myModel.flash_utils.AttnCache import AttnCacheConfig
from opencompass.models.myModel.flash_utils.modeling_internlm2_cached_flash_attn import InternLM2ForCausalLM
import json

def prepare_data_for_eval(data_fp, key, batch_size, seq_len, model_path):
    data = []
    with open(data_fp, 'r', encoding='utf-8') as f:
        for line in f:
            text = json.loads(line[key])
            data.append(text)
    data = data[:batch_size]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    data = tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=seq_len)
    return data

data_fp = './run_flashcache_model/govreport_test.jsonl'
dataset = datasets.load_dataset(data_fp)

data_fp = 'run_flashcache_model/govreport_test.jsonl'
batch_size = 32
model_path = 'models/mossHuawei'