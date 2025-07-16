from agent.rag_chain import load_retriever
from agent.agent_executor import build_agent_executor
from config import Config
from main import init_logger
import logging

def run_agent(message: str, selected_docs: list[str] = [], chat_history: str = "", agent_scratchpad: str = "") -> tuple[str, list]:
  #config = Config()
  retriever = load_retriever(selected_docs)

  logger = logging.getLogger("agent")
  logger.debug(f"[AGENT_INTERFACE-RUN_AGENT Retriever type]\n{type(retriever)}\n")
  logger.debug(f"[AGENT_INTERFACE-RUN_AGENT Retriever attr]\n{dir(retriever)}\n")
  logger.debug(f"[AGENT_INTERFACE-RUN_AGENT Retriever ID]\n{id(retriever)}\n")

  agent = build_agent_executor(logger=logger, retriever_factory=lambda: retriever)
  logger.debug(f"[AGENT_INTERFACE- INPUT KEYS]\n{agent.input_keys}\n")

  result = agent.invoke({
    "input": message,
    #"chat_history": chat_history,
    #"agent_scratchpad": agent_scratchpad
  })
  logger.debug("AGENT_INTERFACE, intermediate_steps dump")
  for action, observation in result["intermediate_steps"]:
    logger.debug(f"Thought/Action/Action Input: {action}")
    logger.debug(f"Observation: {observation}")

  output = result.get("output", "")
  intermediate_steps = result.get("intermediate_steps", [])
  return output, intermediate_steps
