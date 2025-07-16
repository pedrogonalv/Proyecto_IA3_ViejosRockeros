import logging
from langchain_ollama import ChatOllama
from typing import Optional, List
from langchain_core.outputs import ChatResult,ChatGeneration,Generation
from langchain_core.messages import BaseMessage,AIMessage
from pydantic import PrivateAttr, field_validator

class ReActOllamaWrapper(ChatOllama):
  _logger: logging.Logger = PrivateAttr(default=logging.getLogger("agent"))
  _debug_level: int = PrivateAttr(default=2)

  def __call__(self, input, **kwargs):
     return self.invoke(input, **kwargs)

  def invoke(self, input, **kwargs):
    if self._debug_level >= 0:
      self._logger.debug(f"[LLM INVOKE] Received input:\n{input}\n")
      self._logger.debug(f"[LLM INVOKE] Received kwargs:\n{kwargs}\n")

    response = super().invoke(input, **kwargs)

    if self._debug_level >= 0:
      content = response.content if hasattr(response, "content") else str(response)
      self._logger.debug(f"[RAW LLM OUTPUT] Content:\n{content}")

    return response

