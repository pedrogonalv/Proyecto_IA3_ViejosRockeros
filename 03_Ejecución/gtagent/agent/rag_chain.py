from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config
import logging

def load_retriever(selected_docs: list[str] = None):
  logger = logging.getLogger("agent")
  logger.info(f"[RETRIEVER] Loading retriever with doc filter: {selected_docs}")
  config = Config()
  embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model_name)
  vectordb = Chroma(
    persist_directory=config.chroma_persist_dir,
    embedding_function=embeddings,
    collection_name=config.chroma_collection_name
  )
  if selected_docs:
    return vectordb.as_retriever(search_kwargs={
      "k": config.top_k,
      "filter": {"doc_name": {"$in": selected_docs}}
    })
  return vectordb.as_retriever(search_kwargs={"k": config.top_k})

def get_retriever_factory(selected_docs: list[str] = None):
  def retriever():
    return load_retriever(selected_docs)
  return retriever
