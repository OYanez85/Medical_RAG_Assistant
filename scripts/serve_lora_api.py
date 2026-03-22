import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./outputs/qwen25-medical-lora"
SYSTEM_PROMPT = "You are a careful medical assistant. Be accurate, structured, and safety-conscious."

if not os.path.isdir(LORA_PATH):
    raise FileNotFoundError(f"LoRA folder not found: {LORA_PATH}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

app = FastAPI(title="Healthcare LoRA API")

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 160
    temperature: float = 0.2

def build_prompt(user_message: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def extract_assistant_text(decoded: str) -> str:
    marker = "<|im_start|>assistant"
    if marker in decoded:
        decoded = decoded.split(marker, 1)[-1]
    decoded = decoded.replace("<|im_end|>", "").strip()

    # Remove leaked role headers if present
    for prefix in ["system", "user", "assistant"]:
        if decoded.lower().startswith(prefix):
            decoded = decoded[len(prefix):].strip()

    return decoded

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
@torch.inference_mode()
def generate(req: GenerateRequest):
    prompt = build_prompt(req.prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        do_sample=True if req.temperature > 0 else False,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=False)
    response = extract_assistant_text(decoded)

    return {"response": response}
