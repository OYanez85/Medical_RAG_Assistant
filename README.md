---
title: Medical RAG Assistant
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
---

# Medical RAG Assistant

A retrieval-first medical assistant with semantic search, task-aware matching, confidence-based refusal, and emergency safety warnings.

## Features

- Semantic vector retrieval
- Task-aware matching:
  - definition
  - symptoms
  - causes
  - treatment
  - comparison
- Confidence-based refusal
- Emergency keyword detection
- Suggestions for low-confidence cases
- Gradio public interface on Hugging Face Spaces

## Architecture

1. User question enters the system
2. Safety rules check for emergency or high-risk language
3. Query is normalized and aliases are applied
4. Sentence embeddings are computed
5. Top-k semantic matches are retrieved
6. Task-aware reranking improves relevance
7. Low-confidence queries are refused
8. Matching answer is returned with metadata

## Safety Notice

This project is for educational and demonstration purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. If symptoms suggest an emergency, seek urgent medical attention.

## Current Limitations

- Retrieval quality depends on curated dataset coverage
- It does not diagnose individual patients
- It is not connected to live medical literature or hospital systems
- Emergency detection is keyword-based, not clinically exhaustive

## Example Questions

- What is pneumonia?
- What are the symptoms of pneumonia?
- What causes hypertension?
- How is pulmonary embolism treated?
- What is the difference between hypertension and hypotension?
- What is menopause?

## Technical Stack

- Python
- Gradio
- sentence-transformers
- NumPy
- FastAPI-ready backend structure
- Hugging Face Spaces deployment

## Portfolio Summary

This project demonstrates:
- retrieval-augmented QA design
- safe fallback behavior
- task-aware medical information retrieval
- deployable AI application engineering
