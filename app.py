import gradio as gr
from backend.conversational_agent import ConversationalMedicalAgent
from backend.conversation_state import ConversationState

agent = ConversationalMedicalAgent()


def respond(message, chat_history, state_dict):
    state = ConversationState.from_dict(state_dict)

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

    chat_history = chat_history + [{"role": "user", "content": message},
                                   {"role": "assistant", "content": final_answer}]

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
    msg = gr.Textbox(label="Message", placeholder="Ask a medical question...")
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
