from config import Config
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
  config = Config()
  print(f"Testing RAG vector store at: {config.chroma_persist_dir}")
  print(f"Collection: {config.chroma_collection_name}")
  print(f"Embedding model: {config.embedding_model_name}\n")

  try:
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model_name)
    vectordb = Chroma(
      persist_directory=config.chroma_persist_dir,
      embedding_function=embeddings,
      collection_name=config.chroma_collection_name
    )
    store = vectordb._collection

    result = store.get(include=["documents", "metadatas"], limit=10)
    docs = result.get("documents", [])
    metas = result.get("metadatas", [])

    if not docs:
      print("WARNING: No documents found in the vector store.")
    else:
      print(f"Retrieved {len(docs)} documents.\n")
      for i, (doc, meta) in enumerate(zip(docs, metas)):
        print(f"[{i+1}]")
        print("Document:", doc[:150].replace("\n", " ") + "..." if len(doc) > 150 else doc)
        print("Metadata:", meta)
        print("-" * 60)

  except Exception as e:
    print("ERROR: Failed to access vector store:", e)

if __name__ == "__main__":
  main()