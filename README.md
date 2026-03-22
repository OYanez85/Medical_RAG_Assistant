# 🩺 Medical RAG Assistant + LoRA fine-tuned medical LLM

> A hybrid **Retrieval-Augmented Generation (RAG) + LoRA fine-tuned medical LLM** designed for safe, structured, and context-aware medical question answering.

---```

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
```
User Question
↓
Safety Layer (emergency detection)
↓
Query Normalization + Task Detection
↓
───────────────
RAG PATH
───────────────
Embeddings → Retrieval → Reranking → Answer

───────────────
LoRA PATH
───────────────
Prompt → Fine-tuned Model → Generated Answer

 ↓

Final Response
```

---

## 🧪 Example Questions

- What is pneumonia?
- What are the complications of pulmonary embolism?
- How is diabetes treated?
- What causes asthma?
- What is the difference between hypertension and hypotension?

---

## ⚠️ Safety Notice

This project is for **educational and demonstration purposes only**.

It is **NOT**:
- a medical device  
- a diagnostic tool  
- a substitute for professional medical advice  

If symptoms suggest an emergency, seek immediate medical attention.

---

## ⚠️ Known Limitations

- Dataset-driven → limited coverage  
- RAG can retrieve partially relevant content (semantic overlap issues)  
- LoRA model may hallucinate without grounding  
- Not connected to live clinical data (e.g. PubMed, EHR systems)  
- Evaluation shows:
  - repetition in responses  
  - occasional off-topic retrieval (e.g. ORIF leakage)  
  - improvement needed in grounding and citation  

---

## 📊 Evaluation Insights

External evaluation (ReputAgent) highlighted:

- Strong **consistency and safety patterns**
- Weakness in:
  - grounding  
  - negotiation/general reasoning  
  - topic adherence  

👉 This project is actively evolving toward:
- better task-aware routing  
- hybrid RAG + generation  
- improved grounding and citations  

---

## 🛠️ Tech Stack

- Python 3.12
- Gradio
- Transformers (Hugging Face)
- PEFT (LoRA fine-tuning)
- sentence-transformers
- NumPy
- PyTorch
- FastAPI-ready backend

---

## 📁 Project Structure
```
backend/ → RAG engine, reasoning, safety
scripts/ → training, dataset, serving
data/ → QA datasets
outputs/ → LoRA adapters (LFS)
app.py → combined UI (RAG + LoRA)
```

---

## ⚙️ Run Locally

```bash
git clone https://github.com/OYanez85/Medical_RAG_Assistant.git
cd Medical_RAG_Assistant

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python app.py
```

📈 Roadmap
```
 Hybrid RAG + LoRA response synthesis
 Citation-aware generation
 PubMed integration (RAG upgrade)
 Better evaluation pipeline
 Reduced repetition + improved reasoning
 Clinical task classification refinement
```

💼 Portfolio Value
```
This project demonstrates:

End-to-end LLM system design
RAG architecture with task-aware retrieval
LoRA fine-tuning pipeline
Deployment on Hugging Face Spaces
Real-world evaluation and iteration
Safety-aware AI engineering
```

🤝 Author

Oscar Yáñez-Feijoo
Medical SME | AI Engineer | MSc Data Science & AI | Cybersecurity
