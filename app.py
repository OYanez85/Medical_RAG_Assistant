import gradio as gr
from backend.medical_engine import MedicalRAGEngine

engine = MedicalRAGEngine()


def ask_medical_rag(question: str):
    result = engine.ask(question)

    top_matches_text = ""
    if result.top_matches:
        lines = []
        for m in result.top_matches:
            lines.append(
                f"- {m.instruction} | task={m.task} | adjusted={m.adjusted_score:.3f} | raw={m.raw_score:.3f}"
            )
        top_matches_text = "\n".join(lines)

    suggestions_text = "\n".join(result.suggestions) if result.suggestions else ""

    return (
        result.answer,
        result.task,
        f"{result.confidence:.3f}",
        result.matched_question or "",
        str(result.safety_flag),
        result.safety_type or "",
        top_matches_text,
        suggestions_text,
    )


demo = gr.Interface(
    fn=ask_medical_rag,
    inputs=gr.Textbox(
        label="Ask a medical question",
        placeholder="What is pneumonia?"
    ),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Detected task"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Matched question"),
        gr.Textbox(label="Safety flag"),
        gr.Textbox(label="Safety type"),
        gr.Textbox(label="Top matches"),
        gr.Textbox(label="Suggestions"),
    ],
    title="Medical RAG Assistant",
    description="A retrieval-first medical assistant with safety guardrails.",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
