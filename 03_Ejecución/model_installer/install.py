import os
import sys
from tools import (
  is_ollama_running, start_ollama_service,
  model_exists, delete_model, create_model
)
from huggingface_hub import snapshot_download
from config import Config

def download_model_if_needed(config):
  file_path = os.path.join(config.model_local_dir, config.model_filename)
  if not os.path.exists(file_path):
    print("Model not found locally. Downloading from Hugging Face...")
    snapshot_download(config.model_repo_id, local_dir=config.model_local_dir)
  else:
    print("Quantized model already exists locally.")

def main():
  config = Config()

  # Step 1: Check and download quantized model if needed
  download_model_if_needed(config)

  # Step 2: Ensure Ollama service is running
  if not is_ollama_running():
    if not start_ollama_service():
      print("Could not start Ollama service. Aborting.")
      sys.exit(1)

  # Step 3: Install the model in Ollama
  model_name = config.model_name
  quantized_path = os.path.join(config.model_local_dir, config.model_filename)
  
  if model_exists(model_name):
    delete_model(model_name)
  create_model(model_name, quantized_path)

  print("Installation process completed.")

if __name__ == "__main__":
  main()