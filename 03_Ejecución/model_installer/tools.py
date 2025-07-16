import os
import subprocess
import shutil
import sys
import time
from getpass import getpass
from huggingface_hub import (
  snapshot_download, upload_file, create_repo, login
)
from config import Config

def authenticate():
  token=getpass("HF Token: ")
  login(token)
  return token


def clean_and_create_dirs(config):
  # Remove and recreate local directories for a clean state
  dirs_make=[
    config.model_local_dir
  ]
  for d in dirs_make:
    os.makedirs(d, exist_ok=True)
  # Step added: Remove quantized output file if it exists
  quantized_file_path=os.path.join(config.model_output_dir, config.model_filename)
  if os.path.exists(quantized_file_path):
    os.remove(quantized_file_path)

def is_ollama_running():
  try:
    subprocess.run(["ollama", "list"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return True
  except subprocess.CalledProcessError:
    return False

def start_ollama_service():
  try:
    subprocess.Popen(["ollama", "serve"])
    print("Starting Ollama service...")
    for _ in range(10):
      if is_ollama_running():
        print("Ollama service is running.")
        return True
      time.sleep(1)
    print("Could not start Ollama service.")
    return False
  except Exception as e:
    print(f"Error starting Ollama: {e}")
    return False

def model_exists(model_name):
  try:
    result=subprocess.run(["ollama", "list"], check=True, stdout=subprocess.PIPE, text=True)
    return any(line.startswith(model_name+":") for line in result.stdout.splitlines())
  except Exception:
    return False

def delete_model(model_name):
  try:
    subprocess.run(["ollama", "delete", model_name], check=True)
    print(f"Model '{model_name}' has been deleted from Ollama.")
  except Exception as e:
    print(f"Error deleting model '{model_name}': {e}")

def create_model(model_name, model_path):
  modelfile_path=os.path.join(os.path.dirname(model_path), "Modelfile")
  with open(modelfile_path, "w") as f:
    f.write(f"FROM ./{os.path.basename(model_path)}\n")
  cwd=os.getcwd()
  os.chdir(os.path.dirname(model_path))
  try:
    subprocess.run(["ollama", "create", model_name, "-f", "Modelfile"], check=True)
    print(f"Model '{model_name}' has been installed in Ollama.")
  finally:
    os.chdir(cwd)
    try:
      os.remove(modelfile_path)
    except Exception:
      pass

