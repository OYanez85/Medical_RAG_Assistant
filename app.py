import re
import gradio as gr
from backend.conversational_agent import ConversationalMedicalAgent
from backend.conversation_state import ConversationState

agent = ConversationalMedicalAgent()


def count_questions(text: str) -> int:
    parts = re.findall(r'[^?]+\?', text)
    if parts:
        return len(parts)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(lines)


def respond(message, chat_history, state_dict):
    state = ConversationState.from_dict(state_dict)

    qcount = count_questions(message)
    if qcount > 1:
        warning = (
            "Please ask one question at a time so I can preserve the correct topic and follow-up memory.\n\n"
            "Example:\n"
            "1. What is diabetes?\n"
            "2. What are the symptoms?\n"
            "3. How is it treated?\n"
            "4. And complications?"
        )

        chat_history = chat_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": warning},
        ]

        new_state = {
            "history": [t.__dict__ for t in state.history],
            "active_topic": state.active_topic,
            "active_task": state.active_task,
            "last_matched_question": state.last_matched_question,
            "last_confidence": state.last_confidence,
        }

        return chat_history, "", new_state

    result, answer, state, augmented_query = agent.ask(message, state)

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


with gr.Blocks() as demo:
    gr.Markdown("# Medical RAG Assistant v2")

    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(label="Message", placeholder="Ask one medical question at a time...")
    send = gr.Button("Send")
    state = gr.State(value=None)

    send.click(
        fn=respond,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, msg, state],
    )

    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, msg, state],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
