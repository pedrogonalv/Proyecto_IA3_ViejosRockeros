from langchain_core.tools import Tool
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import CallbackManager

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Callable
import textwrap
from config import Config
import logging

config = Config()
embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model_name)

vectordb = Chroma(
  persist_directory=config.chroma_persist_dir,
  embedding_function=embeddings,
  collection_name=config.chroma_collection_name
)
#list_documents is used by the gui
def list_documents(_: str = "", source="tool") -> list[str]:
  logger = logging.getLogger("agent")
  try:
    metadatas = vectordb._collection.get(include=["metadatas"]).get("metadatas", [])
    doc_names = sorted({m["doc_name"] for m in metadatas if "doc_name" in m})
    logger.debug(f"[TOOL list_documents] [{source}] Found documents: {doc_names}")
    return doc_names
  except Exception as e:
    logger.exception(f"[TOOL list_documents] [{source}] Error: {e}")
    return []

def make_list_documents_tool(retriever_factory: Callable[[], BaseRetriever]) -> Callable[[str], str]:
  def list_documents(_: str) -> str:

    logger = logging.getLogger("agent")
    retriever = retriever_factory()

    if retriever is None:
      logger.warning("[TOOL list_documents] No retriever configured.")
      #return "Observation: No retriever configured."
      return "No retriever configured."

    logger.debug(f"[TOOL list_documents (via retriever)] Retriever ID: {id(retriever)}")

    if hasattr(retriever, "search_kwargs"):
      retriever.search_kwargs["k"] = 1000

    r_config = RunnableConfig(tags=["list_documents"])
    docs = retriever.invoke("", config=r_config)

    doc_titles = sorted(set(d.metadata.get("doc_name", "Unknown") for d in docs if "doc_name" in d.metadata))

    logger.info(f"[TOOL list_documents (via retriever)] Found: {doc_titles}")

    if not doc_titles:
      #return "Observation: No documents are currently available in the knowledge base."
      return "No documents are currently available in the knowledge base."
    #return f'Observation: {", ".join(doc_titles)}'
    return f'{", ".join(doc_titles)}'
  return list_documents


def make_search_tool(retriever_factory: Callable[[], BaseRetriever]) -> Callable[[str], str]:
  def search(query: str) -> str:

    logger = logging.getLogger("agent")
    retriever = retriever_factory()

    if retriever is None:
      logger.warning("[TOOL search] No retriever configured.")
      #return "Observation: No retriever configured."
      return "No retriever configured."

    logger.debug(f"[TOOL search] Retriever ID: {id(retriever)}")
    logger.debug(f"[TOOL search] Query: {query}")

    if hasattr(retriever, "search_kwargs"):
      retriever.search_kwargs["k"] = 5

    run_config = RunnableConfig(tags=["search"])

    docs = retriever.invoke(query, config=run_config)

    if not docs:
      #return "Observation: No relevant information found."
      return "No relevant information found."

    result = "\n---\n".join(d.page_content for d in docs)
    logger.info(f"[TOOL search] Result: {result}")
    #return f"Observation: {result}"
    return f"{result}"
  return search

def get_tools(retriever_factory: Callable[[], BaseRetriever], callback_manager: CallbackManager = None):
  list_documents_description = textwrap.dedent("""
Use this tool only when the user asks for the list or names of documents in the knowledge base.

Action: list_documents
Action Input:

Returns a comma-separated list of available document titles.

Do not use this tool to search for technical content or parameters.
  """)
  search_description = textwrap.dedent("""
Use to find specific technical information or answers inside the knowledge base.

Action: search
Action Input: A short question or phrase describing the desired technical detail.

Returns relevant text excerpts from documents.
Do not use parentheses or named arguments.
  """)

  return [
    Tool(
      name="list_documents",
      func=make_list_documents_tool(retriever_factory),
      description=list_documents_description.strip(),
      callbacks=callback_manager
    ),
    Tool(
      name="search",
      func=make_search_tool(retriever_factory),
      description=search_description.strip(),
      callbacks=callback_manager
    )
  ]