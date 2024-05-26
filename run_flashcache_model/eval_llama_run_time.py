import os
import torch
import transformers, datasets
from transformers import AutoTokenizer, AutoConfig
from opencompass.models.myModel.flash_utils.AttnCache import AttnCacheConfig
from opencompass.models.myModel.flash_utils.modeling_internlm2_cached_flash_attn import InternLM2ForCausalLM
import json

def prepare_data_for_eval(data_fp, key, seq_len, model_path):
    data = []
    with open(data_fp, 'r', encoding='utf-8') as f:
        for line in f:
            text = json.loads(line)[key]
            data.append(text)
    
    long_data = data[:1]
    while len(long_data[0].split()) < 31500:
        long_data[0] += ' ' + data[0]

    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    long_data = tokenizer(long_data, return_tensors='pt', padding=True, truncation=True, max_length=seq_len).input_ids
    return long_data

class MeasureGPUTime:
    def __init__(self) -> None:
        self.total_time = 0
        self.record_times = 0

    def __enter__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()  # 记录开始时间

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()  # 记录结束时间
        torch.cuda.synchronize()  # 等待 GPU 完成所有操作
        self.execution_time = self.start_event.elapsed_time(self.end_event) / 1000.0  # 计算执行时间（以秒为单位）
        self.total_time += self.execution_time
        self.record_times += 1
        # print(f"函数的执行时间为: {self.execution_time:.6f} 秒")

    def clear(self):
        self.__init__()

    def avg_record_time(self):
        return self.total_time / self.record_times
         

# seq_lens = [8192]
# num_tokens_to_generate = [2]
seq_lens = [8192, 16000, 32000]
num_tokens_to_generate = [2, 2, 64, 256, 1024]

data_fp = 'run_flashcache_model/govreport_test.jsonl'
# batch_size = 1
# model_path = 'models/mossHuawei'
model_path = '/remote-home/share/models/llama_v2_hf/7b'

test_times = 1

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
USE_CACHED_FLASH_ATTN = True
if USE_CACHED_FLASH_ATTN:
    interlm2_attn_implementation = "cached_flash_attention_2"
    config.attn_cache_config = {
                        "start_size": 4,
                        "recent_size": 2048,
                        "mid_size": 512,
                        "compress_method": "cut-head-suffix",
                        "reserved_dim": 64,
                        "new_decompress_method": True,
                        "max_storage_mid_size": -1,
                        #  "retrieve_method": "none",
                }

    config.attn_cache_config = AttnCacheConfig(**config.attn_cache_config)
    config.attn_implementation = interlm2_attn_implementation
else:
    interlm2_attn_implementation = "flash_attention_2"
    interlm2_attn_implementation = "eager"

model = InternLM2ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True, attn_implementation=interlm2_attn_implementation, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

for ntg in num_tokens_to_generate:
    print('-'*50 + '\n' + f'generate {ntg} tokens', flush=True)
    
    for s in seq_lens:
        # print(f'input {s} tokens', flush=True)
        input_ids = prepare_data_for_eval(data_fp, key='report', seq_len=s, model_path=model_path).cuda()
        # print(f'input {input_ids.shape[-1]} tokens', flush=True)

        mt = MeasureGPUTime()
        for _ in range(test_times):
            with mt:
                with torch.no_grad():
                    if False:
                        generated_ids = model.generate(input_ids, max_new_tokens=ntg, min_new_tokens=ntg, do_sample=False, eos_token_id=2)
                    else:
                        chunk_length = 1024
                        i = 0
                        past_key_values = None

                        while i < input_ids.shape[-1] - 1:
                            input_chunk = input_ids[:, i:min(i + chunk_length, input_ids.shape[-1] - 1)]
                            outputs = model.forward(input_chunk, past_key_values=past_key_values)
                            i += chunk_length
                            past_key_values = outputs.past_key_values
                            # chunk_length = max(1, int(chunk_length * 0.9))  # 确保chunk_length至少为1

                        generated_ids = input_ids[:, -1:].clone().detach()
                        input_ids = input_ids[:, -1:]

                        for _ in range(ntg):
                            outputs = model.forward(input_ids=input_ids, past_key_values=past_key_values)
                            past_key_values = outputs.past_key_values
                            next_token_logits = outputs.logits[:, -1, :]
                            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                            input_ids = next_token_id
                            generated_ids = torch.cat((generated_ids, next_token_id), dim=-1)
                        generated_ids = generated_ids[:, 1:]

                assert generated_ids.shape[-1] == ntg, f'{generated_ids.shape=}, {input_ids.shape=}'
                # print(f'{generated_ids.shape=}, {input_ids.shape=}')
            
        print(f'{mt.avg_record_time():.6f}', flush=True)


        def clean_cache_all_attentions(model):
                        count = 0
                        for name, module in model.named_modules():
                            # if isinstance(module, LlamaCacheAttention):
                                if hasattr(module, 'clean_cache'):
                                    module.clean_cache()
                                    count += 1
                        # print(f"Cleared cache for {count} attention modules.")

        clean_cache_all_attentions(model)