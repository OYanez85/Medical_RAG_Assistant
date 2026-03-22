from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./outputs/qwen25-medical-lora"

if not os.path.isdir(LORA_PATH):
    raise FileNotFoundError(f"LoRA folder not found: {LORA_PATH}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)

prompt = """<|im_start|>system
You are a careful medical assistant. Be accurate and structured.<|im_end|>
<|im_start|>user
What are the complications of pneumonia?<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.2,
        do_sample=True,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
