from agent.agent_executor import build_agent_executor
from agent.ollama_utils import ensure_environment_ready
import logging
import os
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
from agent.rag_chain import get_retriever_factory
from config import Config

def init_logger():
  from config import Config
  config = Config()
  os.makedirs(config.log_dir, exist_ok=True)
  ts = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
  log_path = os.path.join(config.log_dir, f"{ts}.agent.log")
  logging.basicConfig(filename=log_path, level=logging.DEBUG, format="%(message)s")
  for noisy in ["streamlit", "watchdog", "urllib3", "httpx", "asyncio", "PIL","httpcore"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)
  return logging.getLogger("agent")


def format_log_to_messages(intermediate_steps):
  messages = []
  for idx, step in enumerate(intermediate_steps):
    try:
      action, observation = step
      log_text = action.log.strip() if hasattr(action, "log") else str(action)
      obs_text = observation.strip() if observation else ""
      combined = f"{log_text}\nObservation: {obs_text}"
      messages.append(AIMessage(content=combined))
    except Exception as e:
      print(f"[FORMAT SCRATCHPAD ERROR] Step {idx} invalid: {step} → {e}")
  return messages

def format_scratchpad_to_string(messages):
  if not messages:
    return ""
  result = []
  for m in messages:
    if isinstance(m, AIMessage) or isinstance(m, HumanMessage):
      result.append(m.content)
  return "\n".join(result)

def main():
  config=Config
  logger = init_logger()
  logger.info(f"STARTING TAGENT - MODEL: {config.ollama_model}\n")
  ensure_environment_ready(logger)

  retriever_factory = get_retriever_factory(selected_docs=[])
  agent_executor = build_agent_executor(logger, retriever_factory=retriever_factory)
  chat_history = []
  # Ensure agent_scratchpad is formatted correctly
  agent_scratchpad = format_log_to_messages([])  # Initialize with an empty list

  if not isinstance(agent_scratchpad, list) or not all(hasattr(m, "type") for m in agent_scratchpad):
    logger.warning("[SCRATCHPAD WARNING] Invalid messages detected, resetting to empty list.")
    agent_scratchpad = []

  print("Agent is ready. Type your question below. Commands: 'reset_chat' to reset context, 'bye_chat' to exit.")
  while True:
    try:
      query = input(">> ")
      if query.lower() == "bye_chat": break
      if query.lower() == "reset_chat":
        chat_history = []
        print("Chat history has been reset.")
        continue

      logger.debug(
        f"[CHECK] type(agent_scratchpad) BEFORE INVOKE: {type(agent_scratchpad)} | value: {agent_scratchpad}")
      assert isinstance(agent_scratchpad, list), f"Scratchpad malformado: {type(agent_scratchpad)}"

      chat_history_text = "\n".join([
        f"User: {m['content']}" if m['type'] == "human" else f"Assistant: {m['content']}"
        for m in chat_history
      ])

      agent_scratchpad_text = format_scratchpad_to_string(agent_scratchpad)

      logger.info("=== AGENT INVOCATION ===")
      logger.info(f"[USER INPUT] {query}")
      logger.info(f"[CHAT HISTORY]\n{chat_history_text}")
      logger.info(f"[SCRATCHPAD]\n{agent_scratchpad_text}")

      result = agent_executor.invoke({
        "input": query,
        "chat_history": chat_history_text,
        "agent_scratchpad": agent_scratchpad_text
      })
      logger.info(f"[AGENT OUTPUT] {result['output']}")
      print(result["output"])

      #Update scratchpad
      if "intermediate_steps" in result:
        intermediate_steps = result["intermediate_steps"]
      else:
        intermediate_steps = []

      agent_scratchpad = format_log_to_messages(intermediate_steps)
      if not isinstance(agent_scratchpad, list) or not all(hasattr(m, 'type') for m in agent_scratchpad):
        agent_scratchpad = []

      chat_history.append({"type": "human", "content": query})
      chat_history.append({"type": "ai", "content": result["output"]})

      logger.debug(f"[CHAT HISTORY after invoke:]\n{chat_history_text}")
      logger.debug(f"[SCRATCHPAD after invoke:]\n{agent_scratchpad_text}")

    except Exception as e:
      logger.exception(f"[ERROR] {e}")
      print(f"Error: {e}")

if __name__ == "__main__":
  main()
