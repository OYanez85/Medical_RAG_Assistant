# 🩺 Medical RAG Assistant

> A hybrid **Retrieval-Augmented Generation (RAG) + LoRA fine-tuned medical LLM** designed for safe, structured, and context-aware medical question answering.

---

## 🚀 Live Demo

👉 https://huggingface.co/spaces/OscarYanez85/medical-rag-assistant

---

## 🧠 Overview

Medical RAG Assistant started as a **precision-first retrieval system** and has evolved into a **hybrid AI architecture** combining:

- 🔎 **RAG (Retrieval-Augmented Generation)** for grounded, reliable answers  
- 🧬 **LoRA fine-tuned LLM** for fluent, generative responses  
- 🛡️ **Safety-first design** with guardrails and refusal mechanisms  

The system is designed to balance:

> **accuracy + safety + usability**

---

## ✨ Key Features

### 🔎 RAG Engine
- Semantic vector search using **sentence-transformers**
- Hybrid retrieval (semantic + lexical)
- Task-aware classification:
  - definition
  - symptoms
  - causes
  - treatment
  - complications
  - comparison
- Confidence-based refusal
- Query rewriting + alias normalization
- Debug trace (topic, task, confidence)

### 🧬 LoRA Medical LLM
- Fine-tuned on curated medical QA dataset
- Built using:
  - `transformers`
  - `peft (LoRA)`
- Runs locally and via Gradio interface
- Structured prompting with system instructions

### 🛡️ Safety Layer
- Emergency keyword detection
- Controlled answer generation
- Refusal for low-confidence cases
- Designed to avoid hallucinations

### 🌐 Deployment
- Hugging Face Spaces (Gradio UI)
- Local HTTP API (FastAPI-ready)
- GPU-compatible training pipeline (RTX 3070)

---

## 🏗️ Architecture

### High-level pipeline

