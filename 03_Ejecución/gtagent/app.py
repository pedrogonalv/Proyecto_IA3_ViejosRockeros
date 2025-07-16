import streamlit as st
from agent.agent_interface import run_agent
from agent.tools import list_documents
from main import init_logger,format_log_to_messages

from config import Config
from themes.themes import THEMES, apply_theme

config = Config()
logger = init_logger()
theme = THEMES[config.theme]

apply_theme(theme)

if "app_started" not in st.session_state:
  logger.info(f"STARTING GTAGENT - MODEL: {config.ollama_model}\n")
  st.session_state.app_started = True


def format_chat_history_to_string(chat):
  return "\n".join([
    f"User: {msg}" if speaker == "human" else f"Assistant: {msg}"
    for speaker, msg in chat
  ])


st.set_page_config(page_title=config.gui_title, page_icon="🤖", layout="wide")
st.markdown(f"<div class='app-header'>🤖 {config.gui_title}</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1,5])
with col1:
  if st.button("🔄 Reset conversation"):
    logger.info("[GUI-RESET] User pressed reset button.")
    st.session_state.chat_history = []
    st.session_state.agent_scratchpad = []
    st.session_state.user_input_area = ""
    st.rerun()
with col2:
  with st.expander("ℹ️ How to use"):
    st.markdown("""
    - Write your technical question in the input box.
    - Select relevant documents if needed.
    - Click 'Send' to get a response.
    - Use 'Reset' to clear the chat.
    """)

if "chat_history" not in st.session_state:
  st.session_state.chat_history = []
#if "agent_scratchpad" not in st.session_state:
#  st.session_state.agent_scratchpad = []
if "user_input_area" not in st.session_state:
  st.session_state.user_input_area = ""

st.markdown("### Document filter")
all_documents = list_documents(source="streamlit_ui")
if not all_documents:
  st.warning("No documents found in the RAG database.")
  logger.warning("RAG document list is empty.")
selected_docs = st.multiselect("Filter by documents:", all_documents)
st.caption(f"{len(selected_docs)} document(s) selected.")

st.markdown("---")
st.markdown("#### Conversation")

with st.container():
  conversation_html = "<div class='chat-container'>"
  for speaker, msg in st.session_state.chat_history[-config.gui_max_history:]:
    if speaker == "human":
      conversation_html += f"<div style='text-align: right; color: #1f77b4;'><b>🧑‍💻 User:</b> {msg}</div><hr style='margin:4px;'>"
    else:
      conversation_html += f"<div style='text-align: left; color: #2ca02c;'><b>🤖 Agent:</b> {msg}</div><hr style='margin:4px;'>"
  conversation_html += "</div>"
  st.markdown(conversation_html, unsafe_allow_html=True)

st.markdown("#### Ask your question")
with st.form(key="input_form", clear_on_submit=True):
  user_input = st.text_area(
    label="User input",  # etiqueta accesible
    key="user_input_area",
    height=80,
    placeholder="Type your question here...",
    label_visibility="collapsed"  # oculta visualmente pero mantiene accesibilidad
  )
  submit_button = st.form_submit_button("🚀 Send")

if submit_button and user_input.strip():
  chat_history_text = format_chat_history_to_string(st.session_state.chat_history)
  #agent_scratchpad_text = format_scratchpad_to_string(st.session_state.agent_scratchpad)

  example_scratchpad = """Thought: I need to get the list of documents using the list_documents tool.
Action: list_documents
Action Input:
Observation: DocumentA, DocumentB, DocumentC
Thought: I now have the answer.
Final Answer: The available documents are DocumentA, DocumentB, and DocumentC.

Thought: I need to find the maximum voltage supported by the AX5000 drive.
Action: search
Action Input: AX5000 maximum voltage
Observation: The AX5000 drive supports a maximum voltage of 800 V DC.
Thought: I now have the answer.
Final Answer: The AX5000 supports a maximum voltage of 800 V DC.

Thought: I need to list the documents.
Action: Use list_documents  # ← WRONG FORMAT
Observation: ERROR - invalid tool name
Thought: I must follow the correct tool name format.
Action: list_documents
Action Input:
Observation: DocumentX, DocumentY
Thought: I now have the answer.
Final Answer: The available documents are DocumentX and DocumentY.

"""


  #agent_scratchpad_text = clean_scratchpad(st.session_state.agent_scratchpad)

  #if not agent_scratchpad_text.strip():
  #  #agent_scratchpad_text = example_scratchpad
  #  agent_scratchpad_text = ""

  if not selected_docs:
    filtered_docs = all_documents
  else:
    filtered_docs = selected_docs

  #input_with_history = f"Previous conversation:\n{chat_history_text.strip()}\n\nUser: {user_input.strip()}"

  logger.info("=== AGENT INVOCATION PARAMETERS=== start")
  logger.info(f"[INPUT start]\n{user_input}\n[INPUT end]\n")
  #logger.info(f"[INPUT+HISTORY start]\n{input_with_history}\n[INPUT+HISTORY end]\n")
  logger.info(f"[FILTERED DOCS start]\n{filtered_docs}\n[FILTERED DOCS end]\n")
  logger.info("=== AGENT INVOCATION PARAMETERS=== end")

  response, intermediate_steps = run_agent(
    user_input,
    filtered_docs,
    #chat_history=chat_history_text,
    chat_history=chat_history_text,
  )
  logger.info(f"[=== AGENT RESPONSE ===start]\n{response}\n[=== AGENT RESPONSE ===end]\n")

  st.session_state.chat_history.append(("human", user_input))
  st.session_state.chat_history.append(("ai", response))

  logger.info(f"[INTERMEDIATE STEPS after agent run: (start)]\n{intermediate_steps}\n[INTERMEDIATE STEPS after agent run (end)]")
  if intermediate_steps:
    valid_steps = [
      step for step in intermediate_steps
      if hasattr(step, "tool") and step.tool in {"search", "list_documents"}
    ]
    if valid_steps:
      st.session_state.agent_scratchpad = format_log_to_messages(valid_steps)
    else:
      st.session_state.agent_scratchpad = []
  else:
    st.session_state.agent_scratchpad = []

  st.rerun()

