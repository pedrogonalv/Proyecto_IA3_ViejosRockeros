import os
import subprocess
import shutil
import sys
import time
from getpass import getpass
from huggingface_hub import (
  snapshot_download, upload_file, create_repo, login
)
from transformers import AutoTokenizer
from config import Config
import logging
from datetime import datetime

def authenticate(logger):
  token=getpass("HF Token: ")
  login(token)
  logger.info("Authenticated with Huggingface Hub.")
  return token

def create_repo_if_needed(repo_id, token,logger):
  try:
    create_repo(repo_id, private=False, token=token)
  except Exception as e:
    logger.warning(f"Repository {repo_id} already exists or error: {e}")

def clean_and_create_dirs(config,logger):
  # Remove and recreate local directories for a clean state
  dirs_rm=[
    #config.source_local_dir,
    config.gguf_output_dir
  ]
  dirs_make=[
    config.source_local_dir,
    config.gguf_output_dir,
    config.quantized_output_dir
  ]
  for d in dirs_rm:
    if os.path.exists(d):
      shutil.rmtree(d)
      logger.info(f"Removed directory {d}")
  for d in dirs_make:
    os.makedirs(d, exist_ok=True)
    logger.info(f"Created directory {d}")
  quantized_file_path=os.path.join(config.quantized_output_dir, config.quantized_filename)
  if os.path.exists(quantized_file_path):
    os.remove(quantized_file_path)
    logger.info(f"Removed existing quantized file {quantized_file_path}")

def download_source_model(config,logger):
  logger.info("Downloading source model...")
  local_dir = snapshot_download(config.source_repo_id, local_dir=config.source_local_dir)
  logger.info(f"Source model downloaded to {local_dir}")
  return local_dir

def convert_to_gguf(local_dir, config,logger):
  os.makedirs(config.gguf_output_dir, exist_ok=True)
  gguf_path=os.path.join(config.gguf_output_dir, config.gguf_filename)

  if config.override_chat_template:
    script_dir=os.path.dirname(os.path.abspath(__file__))
    chat_template_path=os.path.join(script_dir, config.chat_template_filename)
    if os.path.exists(chat_template_path):
      with open(chat_template_path, "r", encoding="utf-8") as f:
        template_content=f.read()
      logger.info(f"\nChat template loaded from: {chat_template_path}")
      logger.info("LLAMA_CHAT_TEMPLATE content preview:")
      logger.info("-" * 40)
      logger.info(template_content)
      logger.info("-" * 40 + "\n")

      tokenizer_dir = local_dir
      tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
      tokenizer.chat_template = template_content
      tokenizer.save_pretrained(tokenizer_dir)
      logger.info(f"Chat template injected into tokenizer and saved at: {tokenizer_dir}")
    else:
      logger.warning(f"WARNING: chat template not found at: {chat_template_path}")

  cmd = [
    "python", config.convert_script,
    local_dir,
    "--outfile", gguf_path,
    "--outtype", config.gguf_outtype
  ]
  logger.info(f"Running conversion command: {' '.join(cmd)}")
  result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  logger.info(result.stdout)
  if result.stderr:
    logger.error(result.stderr)
  return gguf_path

def upload_gguf(gguf_path, token, config,logger):
  logger.info(f"Uploading GGUF model {gguf_path} to repo {config.gguf_repo_id}...")
  upload_file(
    path_or_fileobj=gguf_path,
    path_in_repo=config.gguf_filename,
    repo_id=config.gguf_repo_id,
    commit_message="Upload GGUF model",
    token=token
  )
  logger.info("GGUF model uploaded.")

def quantize_model(input_path, config,logger):
  os.makedirs(config.quantized_output_dir, exist_ok=True)
  output_path=os.path.join(config.quantized_output_dir, config.quantized_filename)
  cmd=[
    config.llama_cpp_bin,
    input_path,
    output_path,
    config.quantize_type
  ]
  logger.info(f"Running quantization command: {' '.join(cmd)}")
  result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  logger.info(result.stdout)
  if result.stderr:
    logger.error(result.stderr)
  logger.info(f"Quantization completed. Quantized model path: {output_path}")
  return output_path

def upload_quantized_model(quantized_path, token, config,logger):
  logger.info(f"Uploading quantized model {quantized_path} to repo {config.quantized_repo_id}...")
  upload_file(
    path_or_fileobj=quantized_path,
    path_in_repo=config.quantized_filename,
    repo_id=config.quantized_repo_id,
    commit_message="Upload quantized model",
    token=token
  )
  logger.info("Quantized model uploaded.")

def is_ollama_running():
  try:
    subprocess.run(["ollama", "list"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return True
  except subprocess.CalledProcessError:
    return False

def start_ollama_service(logger):
  try:
    subprocess.Popen(["ollama", "serve"])
    logger.info("Starting Ollama service...")
    for _ in range(10):
      if is_ollama_running():
        logger.info("Ollama service is running.")
        return True
      time.sleep(1)
    logger.error("Could not start Ollama service.")
    return False
  except Exception as e:
    logger.error(f"Error starting Ollama: {e}")
    return False

def model_exists(model_name):
  try:
    result=subprocess.run(["ollama", "list"], check=True, stdout=subprocess.PIPE, text=True)
    return any(line.startswith(model_name+":") for line in result.stdout.splitlines())
  except Exception:
    return False

def delete_model(model_name,logger):
  try:
    subprocess.run(["ollama", "delete", model_name], check=True)
    logger.info(f"Model '{model_name}' has been deleted from Ollama.")
  except Exception as e:
    logger.error(f"Error deleting model '{model_name}': {e}")

def create_model(model_name, model_path,logger):
  modelfile_path=os.path.join(os.path.dirname(model_path), "Modelfile")
  with open(modelfile_path, "w") as f:
    f.write(f"FROM ./{os.path.basename(model_path)}\n")
  cwd=os.getcwd()
  os.chdir(os.path.dirname(model_path))
  try:
    subprocess.run(["ollama", "create", model_name, "-f", "Modelfile"], check=True)
    logger.info(f"Model '{model_name}' has been installed in Ollama.")
  finally:
    os.chdir(cwd)
    try:
      os.remove(modelfile_path)
    except Exception:
      pass

def main():
  config=Config()

  os.makedirs(config.log_dir, exist_ok=True)
  now=datetime.now().strftime("%Y%m%d_%H%M%S")
  log_filename=os.path.join(config.log_dir,f"{now}_{config.quantized_model_name}_convert.log")
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
      logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
      logging.StreamHandler(sys.stdout)
    ]
  )
  logger=logging.getLogger()

  logger.info("Starting conversion and quantization process.")

  # Step 1: Clean local directories at the very start
  clean_and_create_dirs(config,logger)

  # Step 2: Authenticate and ensure repo initialization if needed
  token=None
  if config.upload_gguf or config.upload_quantized:
    token=authenticate(logger)
    if config.upload_gguf:
      create_repo_if_needed(config.gguf_repo_id, token,logger)
    if config.upload_quantized:
      create_repo_if_needed(config.quantized_repo_id, token,logger)

  # Step 3: Proceed with the rest of the pipeline
  local_dir=download_source_model(config,logger)
  gguf_path=convert_to_gguf(local_dir, config,logger)

  if config.upload_gguf:
    upload_gguf(gguf_path, token, config,logger)

  quantized_path=quantize_model(gguf_path, config,logger)

  if config.upload_quantized:
    upload_quantized_model(quantized_path, token, config,logger)

  if config.install_in_ollama:
    model_name=config.quantized_model_name
    if not is_ollama_running():
      if not start_ollama_service(logger):
        logger.error("Could not start Ollama service. Aborting.")
        sys.exit(1)
    if model_exists(model_name):
      delete_model(model_name,logger)
    create_model(model_name, quantized_path,logger)

  logger.info("Process completed.")

if __name__=="__main__":
  main()