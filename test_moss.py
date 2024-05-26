from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

model_path = '/remote-home/share/storage/zyyin/moss2Huawei'
# model_path = '/remote-home/share/models/moss2Huawei'
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, device_map='cuda', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

text = "The capital of France is"
input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
output = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0]))