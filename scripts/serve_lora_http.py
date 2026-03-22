import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./outputs/qwen25-medical-lora"

SYSTEM_PROMPT = "You are a careful medical assistant. Be accurate, structured, and safety-conscious."

if not os.path.isdir(LORA_PATH):
    raise FileNotFoundError(f"LoRA folder not found: {LORA_PATH}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

def build_prompt(user_message: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

@torch.inference_mode()
def generate_response(message, max_new_tokens=160, temperature=0.2):
    prompt = build_prompt(message)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        do_sample=True if float(temperature) > 0 else False,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    if "assistant" in decoded:
        decoded = decoded.split("assistant", 1)[-1].strip()

    return decoded

demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Textbox(label="Medical question", lines=4, placeholder="Ask a medical question..."),
        gr.Slider(32, 256, value=160, step=8, label="Max new tokens"),
        gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Model answer", lines=10),
    title="Healthcare LoRA Demo",
    description="Local HTTP demo for your fine-tuned Qwen2.5 medical adapter",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
