import gradio as gr
import httpx
import json
import os
from typing import List               

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")


class StreamingMarkdownRenderer:
    def __init__(self):
        self.container_stack: List[str] = []
        self.context: str = 'key'
        self.in_string: bool = False
        self.is_escaped: bool = False
        self.key_buffer: str = ""
        self.first_item_in_container: bool = True

    def _reset_state(self):
        self.container_stack.clear(); self.context = 'key'; self.in_string = False
        self.is_escaped = False; self.key_buffer = ""; self.first_item_in_container = True

    def _format_key(self, key: str) -> str:
        return ' '.join(word.capitalize() for word in key.split('_'))

    def process_event(self, event: dict):
        chunk = event.get("content", "")
        for char in chunk:
            yield from self._process_char(char)

    def _process_char(self, char: str):
        if self.is_escaped:
            yield char
            self.is_escaped = False
            return
        if char == '\\':
            self.is_escaped = True
            yield char
            return
        if char == '"':
            self.in_string = not self.in_string
            if self.in_string and self.context == 'value' and self.container_stack and self.container_stack[-1] == '[':
                if not self.first_item_in_container:
                    yield "\n"
                indent = "  " * self.container_stack.count('{')
                yield f"{indent}- "
                self.first_item_in_container = False
            return
        if self.in_string:
            if self.context == 'key':
                self.key_buffer += char
            else:
                yield char
        else:
            yield from self._process_structural_char(char)

    def _process_structural_char(self, char: str):
        if char.isspace(): return
        if char in '{[':
            if self.container_stack and self.container_stack[-1] == '[':
                if not self.first_item_in_container: yield "\n"
                indent = "  " * self.container_stack.count('{'); yield f"{indent}- "
            if char == '[' and self.context == 'value': yield "\n"
            self.container_stack.append(char); self.context = 'key' if char == '{' else 'value'; self.first_item_in_container = True
        elif char in '}]':
            if self.container_stack: self.container_stack.pop()
            yield "\n\n"
        elif char == ':':
            self.context = 'value'; self.first_item_in_container = True; key = self.key_buffer.strip(); self.key_buffer = ""
            yield from self._format_value_prefix(key)
        elif char == ',':
            self.context = 'key' if self.container_stack and self.container_stack[-1] == '{' else 'value'; self.first_item_in_container = False

    def _format_value_prefix(self, key: str):
        if key in ('main_idea', 'title'): yield "### "
        elif key in ('text', 'points', 'details', 'description'): yield " " if self.container_stack.count('{') > 0 else "\n"
        else:
            is_top_level = self.container_stack.count('{') == 1
            if is_top_level: yield f"### {self._format_key(key)}\n"
            else:
                indent = "  " * (self.container_stack.count('{') - 1); yield f"\n{indent}- **{self._format_key(key)}:** "


async def manual_chat(message: str, history: list):
    """
    Handles the core logic of a single chat round.
    It appends messages and streams the response from the RAG API.
    """
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    yield history

    renderer = StreamingMarkdownRenderer()
    renderer._reset_state()
    response_buffer = ""

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{RAG_API_URL}/ask", json={"question": message}, timeout=None) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        if event.get("type") == "stream_end":
                            break
                        if event.get("type") == "error":
                            error_content = event.get('content', 'Unknown error')
                            response_buffer += f"\n\n**Error:** {error_content}"
                            history[-1]["content"] = response_buffer
                            yield history
                            continue
                        
                        for md_chunk in renderer.process_event(event):
                            response_buffer += md_chunk
                            history[-1]["content"] = response_buffer
                            yield history
    except httpx.HTTPStatusError as e:
        history[-1]["content"] = f"**Error connecting to RAG API:** {e.response.status_code} - Check if the service is running."
    except httpx.RequestError as e:
        history[-1]["content"] = f"**Network Error:** Could not connect to the RAG API at {RAG_API_URL}."
    except Exception as e:
        history[-1]["content"] = f"**An unexpected error occurred:** {e}"
    finally:
        
        yield history


custom_css = """

.gradio-container {
  background: #18181b !important;
  font-family: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial !important;
  color: #f4f4f5 !important;
}

.gr-chatbot {
  background: #27272a !important;
  border-radius: 14px !important;
  box-shadow: 0 6px 18px rgba(0,0,0,0.5) !important;
  padding: 18px !important;
  border: 1px solid #3f3f46 !important;
}

.gr-message {
  border-radius: 12px !important;
  padding: 12px 16px !important;
  font-size: 15px !important;
  line-height: 1.5 !important;
  max-width: 75% !important;
  margin-bottom: 6px !important;
}
.gr-message.assistant {
  background: #3f3f46 !important;
  color: #fafafa !important;
}
.gr-message.user {
  background: #4f46e5 !important;
  color: #ffffff !important;
  font-weight: 500 !important;
  align-self: flex-end !important;
}

.gr-textbox textarea {
  background: #18181b !important;
  border: 1px solid #3f3f46 !important;
  border-radius: 10px !important;
  padding: 12px 14px !important;
  font-size: 15px !important;
  color: #f4f4f5 !important;
  resize: none !important;
  outline: none !important;
  transition: border 0.2s ease, background 0.2s ease !important;
}
.gr-textbox textarea:focus {
  border: 1px solid #6366f1 !important;
  background: #1f1f23 !important;
}

.gradio-container .gr-button {
  border-radius: 10px !important;
  padding: 10px 14px !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  background: #4f46e5 !important;
  color: #fff !important;
  border: none !important;
  transition: background 0.2s ease !important;
}
.gradio-container .gr-button:hover {
  background: #3730a3 !important;
}

.faq-chip {
  background: #27272a !important;
  border: 1px solid #3f3f46 !important;
  padding: 10px 16px !important;
  border-radius: 999px !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  color: #f4f4f5 !important;
  cursor: pointer !important;
  transition: all 0.15s ease !important;
}
.faq-chip:hover {
  background: #4f46e5 !important;
  color: #fff !important;
  transform: translateY(-2px) !important;
}
"""
welcome_message = [{"role": "assistant", "content": "Welcome! I am a question-answering assistant. How can I help you today?"}]
faq_questions = [
    "What subjects do you offer courses in?",
    "Can you give me details about the Full-Stack Web Development course?",
    "Compare the Python course to the Data Science bootcamp.",
    "What are the pricing options?",
    "Tell me about student support.",
    "Do I get a certificate after completion?",
]

with gr.Blocks(css=custom_css, theme=gr.themes.Default(primary_hue="indigo")) as demo:
    gr.Markdown("<h1><center>Adaptive RAG Engine (Decoupled)</center></h1>")

    chatbot_ui = gr.Chatbot(
        value=welcome_message, 
        height=520, 
        elem_id="chatbot",
        type="messages" 
    )

    with gr.Row():
        textbox = gr.Textbox(
            placeholder="Ask me anything about the provided knowledge...",
            container=False, 
            scale=7,
        )
        submit_button = gr.Button("Send", variant="primary", scale=1)

    
    gr.Examples(
        examples=faq_questions,
        inputs=textbox,
        label="Example Questions"
    )

    async def submit_and_clear(message: str, history: list):
        
        async for hist in manual_chat(message, history):
            yield hist, ""

    
    textbox.submit(
        fn=submit_and_clear,
        inputs=[textbox, chatbot_ui],
        outputs=[chatbot_ui, textbox]
    )
    submit_button.click(
        fn=submit_and_clear,
        inputs=[textbox, chatbot_ui],
        outputs=[chatbot_ui, textbox]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)