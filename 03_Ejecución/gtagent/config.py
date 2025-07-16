import os

class Config:
  # Ollama
  #ollama_model = "MGiver-Nous-Hermes-2-qlora-merged-gguf-q4"
  #ollama_model = "Nous-Hermes-2-Mistral-7B-DPO-gguf-q4"
  #ollama_model = "MGiver-Mistral-7B-i-qlora-merged-gguf-q4"
  #ollama_model = "Mistral-7B-Technical-Tutorial-Summarization-QLoRA-gguf-q4"
  #ollama_model = "Mistral-7B-Instruct-v0.3-gguf-q4"
  #ollama_model = "MGiver-2-Mistral-7B-i-qlora-merged-gguf-q4"
  #ollama_model = "MGiver-3-Mistral-7B-i-qlora-merged-gguf-q4"
  ollama_model = "MGiver-4-Mistral-7B-i-qlora-merged-gguf-q4"

  ollama_host = "http://localhost:11434"
  ollama_autostart = True

  # Chroma DB
  chroma_persist_dir = "./data/chroma_db"
  chroma_collection_name = "pdf_qa"
  embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"

  # Parámetros del agente
  temperature = 0.3
  top_k = 5
  max_iterations = 8

  # Debugging
  debug_level = 3  # 0 = nothing, 1 = basic, 2 = deeper, 3 = deepest
  log_dir = "./logs"
  save_context = True
  context_dir = "./contexts"
# GUI (Streamlit)
  gui_title = "GTagent - Asistente RAG"
  gui_max_history = 50
  theme = "factory"