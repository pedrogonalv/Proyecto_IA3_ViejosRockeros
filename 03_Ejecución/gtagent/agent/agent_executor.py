from langchain.agents import initialize_agent, AgentType
#from langchain.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManager
from langchain.memory import ConversationBufferMemory

from agent.ollama_wrapper import ReActOllamaWrapper
from langchain.callbacks.base import BaseCallbackHandler
from agent.tools import get_tools
import inspect
from config import Config
import agent.prompts as prompt_templates
import os

class LoggingCallbackHandler(BaseCallbackHandler):
  def __init__(self, logger, debug_level, context_prefix=None, save_context=False):
    super().__init__()
    self.logger = logger
    self.debug_level = debug_level
    self.context_prefix = context_prefix
    self.save_context = save_context
    self.raw_steps = 0
    self.steps = 0
    self.invalid_tool_streak = 0
    self.max_invalid_tool_streak = 3
    self.memory=None

  def _reset_scratchpad(self):
    self.logger.debug(f"[LoggingCallbackHandler._reset_scrathpad]. Memory is going to be cleared and written with reactive message.")
    if hasattr(self, "memory") and self.memory is not None:
      self.memory.clear()
      self.memory.chat_memory.add_user_message("You are not using valid tools.")
      self.memory.chat_memory.add_ai_message(
        "I apologize. I will only use valid tools from now on: 'list_documents' and 'search'."
      )

  def on_tool_start(self, serialized, input_str, **kwargs):
    self.steps += 1
    if self.debug_level >= 2:
      self.logger.debug(f"[TOOL START] Step {self.steps} | {serialized['name']} | input: {input_str}")
      self.logger.debug(f"[CONTEXT LENGTH] {len(input_str)} caracteres")
    if self.debug_level >= 3:
      self.logger.debug(f"[CONTEXT CONTENT] >>>\n{input_str}\n<<<")
    if self.save_context and self.context_prefix:
      fname = f"{self.context_prefix}_step_{self.steps:02d}.txt"
      with open(fname, "w") as f:
        f.write(input_str)

  def on_tool_end(self, output, **kwargs):
    if self.debug_level >= 2:
      self.logger.debug(f"[TOOL END] output: {output[:500]}...")

  def on_agent_action(self, action, **kwargs):
    self.raw_steps += 1
    if self.debug_level >= 1:
      self.logger.debug(f"[AGENT ACTION] step {self.raw_steps}:\n{action.log}\n")
      self.logger.debug(f"[AGENT ACTION] Decoded tooling data: tool={action.tool}, input={action.tool_input}\n")
    if action.tool not in ["search", "list_documents"]:
      self.invalid_tool_streak += 1
      self.logger.warning(f"[AGENT ACTION][INVALID TOOL] '{action.tool}' is not a registered tool.\n")
      if self.invalid_tool_streak >= self.max_invalid_tool_streak:
        self.logger.error("[AGENT ACTION] Too many invalid actions. Resetting scratchpad and inserting feedback.")
        self._reset_scratchpad()
        raise ValueError("Too many invalid actions in a row, scratchpad reset.")
      raise ValueError(f"Invalid tool: {action.tool}")
    else:
      self.invalid_tool_streak=0



  def on_agent_finish(self, finish, **kwargs):
    if self.debug_level >= 1:
      self.logger.debug(f"[AGENT FINISH] return_values={finish.return_values} log=\"{finish.log}\"")

  def on_chain_error(self, error, **kwargs):
    if self.debug_level >= 1:
      self.logger.error(f"[CHAIN ERROR] {error}")

  def on_tool_error(self, error, **kwargs):
    if self.debug_level >= 1:
      self.logger.error(f"[TOOL ERROR] {error}")

  def on_llm_error(self, error, **kwargs):
    if self.debug_level >= 1:
      self.logger.error(f"[LLM ERROR] {error}")
  def on_llm_start(self, serialized, prompts, **kwargs):
    if self.debug_level >= 3:
      for i,prompt in enumerate(prompts):
        self.logger.debug(f"[LLM PROMPT - {i}] >>>\n{prompt}\n<<<")

def preload_memory_examples(memory):
  memory.chat_memory.add_user_message("What documents are available?")
  memory.chat_memory.add_ai_message('''
Thought: I need to list the documents
Action: list_documents
Action Input:
Observation: doc1.pdf, doc2.pdf
Thought: I now know the final answer
Final Answer: The available documents are doc1.pdf and doc2.pdf."
'''
  )

  memory.chat_memory.add_user_message("What is the best football club in the world?")
  memory.chat_memory.add_ai_message('''
Thought: To find the best football club in the world I need to search at the database
Action: search
Action Input: best football club in the world
Observation: The best football club in the world is Real Madrid, leading the UCL championships ranking.
Thought: I now know the final answer.
Final Answer: The best football club is Real Madrid, so far is the one that has won the biggest amount of UCL championships.
  '''
  )

def build_agent_executor(logger=None, retriever_factory=None):
  config = Config()
  if retriever_factory is None:
    raise ValueError("retriever_factory must be provided to build_agent_executor")

  context_prefix = None
  if config.save_context and logger:
    for handler in logger.handlers:
      if hasattr(handler, "baseFilename"):
        log_path = handler.baseFilename
        ts = os.path.splitext(os.path.basename(log_path))[0]
        context_prefix = os.path.join(config.context_dir, ts)

  memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output"
  )
  preload_memory_examples(memory)

  cb_handler=LoggingCallbackHandler(logger, config.debug_level, context_prefix, config.save_context)
  cb_handler.memory = memory
  callbacks = [cb_handler] if logger else []
  callback_manager = CallbackManager(callbacks)

  llm = ReActOllamaWrapper(
    model=config.ollama_model,
    temperature=config.temperature,
    callbacks=callbacks
  )
  object.__setattr__(llm, "_logger", logger)
  object.__setattr__(llm, "_debug_level", config.debug_level)


  tools = get_tools(retriever_factory=retriever_factory, callback_manager=callback_manager)

  if logger:
    for t in tools:
      logger.info(f"[TOOL REGISTERED] {t.name} | func={repr(t.func)} | type={type(t.func)}")
      logger.info(f"[TOOL FUNC INSPECT] {t.name} accepts: {inspect.signature(t.func)}")
  assert all(hasattr(t.func, "__call__") for t in tools), "All tools must have callable functions"

  system_message_text = prompt_templates.system_message_text_template_9

  #prompt = PromptTemplate.from_template(
  #  template=system_message_text,
  #  #input_variables=["input"]
  #)

  agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=config.debug_level >= 2,
    return_intermediate_steps=True,
    max_iterations=config.max_iterations,
    callbacks=callbacks,
    memory=memory,
    handle_parsing_errors="retry",
    #agent_kwargs={
    #  "prompt": prompt,
    #}
  )

  return agent_executor

