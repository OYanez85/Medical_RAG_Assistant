from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_FILE = "data/llm_train.jsonl"
VAL_FILE = "data/llm_val.jsonl"
OUTPUT_DIR = "./outputs/qwen25-medical-lora"

MAX_LENGTH = 384


def format_chat(example):
    messages = example["messages"]
    text = ""
    for msg in messages:
        role = msg["role"].strip().lower()
        content = msg["content"].strip()

        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"

    return {"text": text}


def tokenize(example, tokenizer):
    out = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    out["labels"] = out["input_ids"].copy()
    return out


def main():
    os.makedirs("./outputs", exist_ok=True)

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VAL_FILE,
        }
    )

    dataset = dataset.map(format_chat)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    keep_cols = {"input_ids", "attention_mask", "labels"}
    dataset = dataset.remove_columns(
        [c for c in dataset["train"].column_names if c not in keep_cols]
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",
        eval_strategy="steps",
        optim="paged_adamw_8bit",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
