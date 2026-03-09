import gradio as gr
from backend.medical_engine import MedicalRAGEngine
from backend.logger import log_query
from pathlib import Path
import json
from datetime import datetime

engine = MedicalRAGEngine()
BASE = Path(__file__).resolve().parent
FEEDBACK_LOG = BASE / "logs" / "feedback.jsonl"
FEEDBACK_LOG.parent.mkdir(exist_ok=True)


def ask_medical_rag(question):
    result = engine.ask(question)

    details = []
    details.append(f"Detected task: {result.task}")
    details.append(f"Confidence: {result.confidence:.3f}")
    details.append(f"Matched question: {result.matched_question or 'N/A'}")
    details.append(f"Safety flag: {result.safety_flag}")
    details.append(f"Safety type: {result.safety_type or 'none'}")

    if result.top_matches:
        details.append("Top matches:")
        for m in result.top_matches[:5]:
            details.append(
                f"- {m.instruction} | task={m.task} | adjusted={m.adjusted_score:.3f} | raw={m.raw_score:.3f}"
            )

    if result.suggestions:
        details.append("Suggestions:")
        for s in result.suggestions:
            details.append(f"- {s}")

    answer_text = result.answer
    details_text = "\n".join(details)

    log_query({
        "question": result.question,
        "normalized_question": result.normalized_question,
        "task": result.task,
        "safety_flag": result.safety_flag,
        "safety_type": result.safety_type,
        "confidence": result.confidence,
        "matched_question": result.matched_question,
        "answer": result.answer,
        "suggestions": result.suggestions,
        "source": "gradio_feedback_ui",
    })

    state = {
        "question": result.question,
        "answer": result.answer,
        "matched_question": result.matched_question,
        "task": result.task,
        "confidence": result.confidence,
        "timestamp": datetime.utcnow().isoformat(),
    }

    return answer_text, details_text, state


def save_feedback(state, label):
    if not state:
        return "No result to rate yet."

    record = dict(state)
    record["feedback"] = label
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return f"Saved feedback: {label}"


with gr.Blocks() as demo:
    gr.Markdown("# Medical RAG Assistant")
    gr.Markdown("A retrieval-first medical assistant with safety guardrails.")

    question = gr.Textbox(label="Ask a medical question", placeholder="What is pneumonia?")
    submit = gr.Button("Submit")

    answer = gr.Textbox(label="Answer")
    details = gr.Textbox(label="Details", lines=12)
    feedback_status = gr.Textbox(label="Feedback status")

    state = gr.State()

    with gr.Row():
        thumbs_up = gr.Button("👍 Correct")
        thumbs_down = gr.Button("👎 Incorrect")

    submit.click(
        fn=ask_medical_rag,
        inputs=[question],
        outputs=[answer, details, state]
    )

    thumbs_up.click(
        fn=lambda s: save_feedback(s, "correct"),
        inputs=[state],
        outputs=[feedback_status]
    )

    thumbs_down.click(
        fn=lambda s: save_feedback(s, "incorrect"),
        inputs=[state],
        outputs=[feedback_status]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
