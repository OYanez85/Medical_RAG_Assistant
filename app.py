import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from backend.conversational_agent import ConversationalMedicalAgent
from backend.conversation_state import ConversationState

# -------------------------
# RAG SETUP
# -------------------------
rag_agent = ConversationalMedicalAgent()

def rag_respond(message, chat_history, state_dict):
    state = ConversationState.from_dict(state_dict)

    result, answer, state, augmented_query = rag_agent.ask(message, state)

    debug = (
        f"\n\n---\n"
        f"Rewritten query: {augmented_query}\n"
        f"Active topic: {state.active_topic}\n"
        f"Matched question: {result.matched_question or 'N/A'}\n"
        f"Detected task: {result.task}\n"
        f"Confidence: {result.confidence:.3f}\n"
    )

    final_answer = answer + debug

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": final_answer},
    ]

    new_state = {
        "history": [t.__dict__ for t in state.history],
        "active_topic": state.active_topic,
        "active_task": state.active_task,
        "last_matched_question": state.last_matched_question,
        "last_confidence": state.last_confidence,
    }

    return chat_history, "", new_state


# -------------------------
# LORA SETUP
# -------------------------
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./outputs/qwen25-medical-lora"
SYSTEM_PROMPT = "You are a careful medical assistant. Be accurate, structured, and safety-conscious."

print("Loading LoRA tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("Loading LoRA base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapter...")
lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)
lora_model.eval()

def build_prompt(user_message: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def extract_assistant_text(decoded: str) -> str:
    marker = "<|im_start|>assistant\n"
    if marker in decoded:
        decoded = decoded.split(marker, 1)[-1]
    decoded = decoded.replace("<|im_end|>", "").strip()
    return decoded

@torch.inference_mode()
def generate_lora_response(message, max_new_tokens=160, temperature=0.2):
    prompt = build_prompt(message)
    inputs = tokenizer(prompt, return_tensors="pt").to(lora_model.device)

    output = lora_model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        do_sample=True if float(temperature) > 0 else False,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=False)
    return extract_assistant_text(decoded)


# -------------------------
# UI
# -------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Healthcare AI Demo")
    gr.Markdown("Compare retrieval-grounded medical QA with a fine-tuned LoRA medical model.")

    with gr.Tabs():
        with gr.Tab("RAG Assistant"):
            chatbot = gr.Chatbot(type="messages")
            msg = gr.Textbox(label="Message", placeholder="Ask one medical question at a time...")
            send = gr.Button("Send")
            rag_state = gr.State(value=None)

            send.click(
                fn=rag_respond,
                inputs=[msg, chatbot, rag_state],
                outputs=[chatbot, msg, rag_state],
            )

            msg.submit(
                fn=rag_respond,
                inputs=[msg, chatbot, rag_state],
                outputs=[chatbot, msg, rag_state],
            )

        with gr.Tab("LoRA Assistant"):
            gr.Interface(
                fn=generate_lora_response,
                inputs=[
                    gr.Textbox(label="Medical question", lines=4, placeholder="Ask a medical question..."),
                    gr.Slider(32, 256, value=160, step=8, label="Max new tokens"),
                    gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature"),
                ],
                outputs=gr.Textbox(label="Model answer", lines=10),
                allow_flagging="never",
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
