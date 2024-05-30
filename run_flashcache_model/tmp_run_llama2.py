from transformers import AutoTokenizer, AutoConfig
import torch
from opencompass.models.myModel.flash_utils_v2.AttnCache import AttnCacheConfig

from opencompass.models.myModel.flash_utils_v2.modeling_internlm2_cached_flash_attn import InternLM2ForCausalLM
from opencompass.models.myModel.flash_utils_v2.modeling_llama_cached_flash_attn import LlamaForCausalLM

model_path = '/remote-home/share/models/llama_v2_hf/7b'

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)


USE_CACHED_FLASH_ATTN = True
if USE_CACHED_FLASH_ATTN:
    interlm2_attn_implementation = "cached_flash_attention_2"
    config.attn_cache_config = {
                        "start_size": 4,
                        "recent_size": 64,
                        "mid_size": 32,

                        "key_compress_method": 'incrementalpca',
                        "key_reserved_dim": 64,
                        "key_compress_split_head": False,
                        
                        "value_compress_method": "none",
                        "value_reserved_dim": 4096,
                        "value_compress_split_head": False,
                }

    config.attn_cache_config = AttnCacheConfig(**config.attn_cache_config)
    config.attn_implementation = interlm2_attn_implementation
else:
    interlm2_attn_implementation = "flash_attention_2"


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True, attn_implementation=interlm2_attn_implementation, config=config)

prompt = "User: Please tell me about China. Answer:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

max_new_tokens = 256

with torch.no_grad():
    if not USE_CACHED_FLASH_ATTN:
        outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        # print("outputs.shape: ", outputs.shape)
        # print("input_ids.shape: ", input_ids.shape)
        outputs = outputs[..., input_ids.shape[-1]:]
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    else:
        chunk_length = 3
        i = 0

        while i < input_ids.shape[-1] - 1:
            input_chunk = input_ids[:, i:min(i + chunk_length, input_ids.shape[-1] - 1)]
            model.forward(input_chunk)
            i += chunk_length
            # chunk_length = max(1, int(chunk_length * 0.9))  # 确保chunk_length至少为1

        generated_ids = input_ids[:, -1:].clone().detach()
        input_ids = input_ids[:, -1:]
        past_key_values = None

        for _ in range(max_new_tokens):
            outputs = model.forward(input_ids=input_ids, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = next_token_id
            generated_ids = torch.cat((generated_ids, next_token_id), dim=-1)
            if next_token_id == tokenizer.eos_token_id or next_token_id == tokenizer.pad_token_id:
                break

        # output_tokens = self.model.generate(input_ids[:, -1:], max_new_tokens=max_out_len, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0][1:], skip_special_tokens=False)

print(generated_text)
print("attn_implementation:", model.config._attn_implementation)
