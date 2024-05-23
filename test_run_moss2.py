from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = '/remote-home/share/models/mossHuawei/'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map='cuda', trust_remote_code=True)

prompt = "你好，我是"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
output = model.generate(input_ids, max_length=512, do_sample=True, top_p=0.7, top_k=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))

model(1)