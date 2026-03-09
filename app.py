import gradio as gr
from backend.medical_engine import MedicalRAGEngine
from backend.logger import log_query

engine = MedicalRAGEngine()


def ask_medical_rag(message, history):
    result = engine.ask(message)

    details = []
    details.append(f"**Detected task:** {result.task}")
    details.append(f"**Confidence:** {result.confidence:.3f}")
    details.append(f"**Matched question:** {result.matched_question or 'N/A'}")
    details.append(f"**Safety flag:** {result.safety_flag}")
    details.append(f"**Safety type:** {result.safety_type or 'none'}")

    if result.top_matches:
        details.append("**Top matches:**")
        for m in result.top_matches[:5]:
            details.append(
                f"- {m.instruction} | task={m.task} | adjusted={m.adjusted_score:.3f} | raw={m.raw_score:.3f}"
            )

    if result.suggestions:
        details.append("**Suggestions:**")
        for s in result.suggestions:
            details.append(f"- {s}")

    details_text = "\n".join(details)
    final_answer = f"{result.answer}\n\n---\n{details_text}"

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
        "source": "gradio_chat",
    })

    return final_answer


demo = gr.ChatInterface(
    fn=ask_medical_rag,
    title="Medical RAG Assistant",
    description="A retrieval-first medical assistant with safety guardrails.",
    textbox=gr.Textbox(
        placeholder="Ask a medical question, e.g. What is pneumonia?",
        label="Question"
    ),
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
