import subprocess
import requests
import time
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import Config

def is_ollama_running(config: Config) -> bool:
  try:
    r = requests.get(f"{config.ollama_host}/api/tags", timeout=2)
    return r.status_code == 200
  except:
    return False

def start_ollama_model(config: Config, logger=None):
  try:
    subprocess.Popen(["ollama", "run", config.ollama_model], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if logger: logger.info(f"[OLLAMA] Lanzando modelo '{config.ollama_model}' con ollama run ...")
    time.sleep(5)
  except Exception as e:
    if logger: logger.error(f"[OLLAMA] Error al intentar lanzar ollama: {e}")

def check_chroma_ready(config: Config, logger=None):
  try:
    if not os.path.exists(config.chroma_persist_dir):
      raise FileNotFoundError(f"No se encontró el directorio {config.chroma_persist_dir}")
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model_name)
    vectordb = Chroma(
      persist_directory=config.chroma_persist_dir,
      embedding_function=embeddings,
      collection_name=config.chroma_collection_name
    )
    vectordb._collection.count()
    if logger: logger.info("[CHROMA] Base de datos vectorial cargada correctamente.")
  except Exception as e:
    if logger: logger.error(f"[CHROMA] Error al acceder a la base de datos: {e}")
    raise

def ensure_environment_ready(logger=None):
  config = Config()
  if is_ollama_running(config):
    if logger: logger.info("[OLLAMA] Servicio activo.")
  elif config.ollama_autostart:
    if logger: logger.warning("[OLLAMA] No activo. Intentando lanzar el modelo...")
    start_ollama_model(config, logger)
  else:
    if logger: logger.error("[OLLAMA] No se detecta el servicio y autostart está desactivado.")
    raise RuntimeError("Ollama no disponible y no se puede lanzar automáticamente.")

  check_chroma_ready(config, logger)
