import os
import sys
import shutil
from getpass import getpass
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from huggingface_hub import login, create_repo, upload_folder
from config import Config
import logging
from datetime import datetime

config=Config()



os.environ["HF_HOME"] = "/workspace/hugging_face_res"

def authenticate(logger):
  logger.info("HF authentication.")
  token = getpass("Token HF: ")
  login(token)
  return token

def create_repo_if_needed(repo_id, token,logger):
  logger.info(f"Creaging repo {repo_id}")
  try:
    create_repo(repo_id, private=config.merged_repo_private, token=token)
  except Exception as e:
    print()
    logger.warning(f"Repository already exists or error:\n {e}")

def merge_models(logger):
  logger.info(f"Merging")
  peft_config = PeftConfig.from_pretrained(config.adapter_repo_id)
  logger.info(f"Loading base model")
  base_model = AutoModelForCausalLM.from_pretrained(
    config.source_repo_id,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto"
  )

  logger.info(f"Loading tokenizer")
  tokenizer = AutoTokenizer.from_pretrained(config.source_repo_id, use_fast=False, trust_remote_code=True)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  model = PeftModel.from_pretrained(base_model, config.adapter_repo_id)
  logger.info(f"Getting merged model.")
  merged_model = model.merge_and_unload()
  return merged_model, tokenizer

def save_and_upload(model, tokenizer, token,logger):
  logger.info("Uploading merged model")
  model.save_pretrained(config.merged_output_dir, safe_serialization=True, max_shard_size="5GB")
  tokenizer.save_pretrained(config.merged_output_dir)
  upload_folder(
    repo_id=config.merged_repo_id,
    folder_path=config.merged_output_dir,
    commit_message="Upload merged model",
    token=token
  )

def main():
  log_dir = config.log_dir
  os.makedirs(log_dir, exist_ok=True)

  now = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_filename = os.path.join(config.log_dir, f"{now}_{config.adapter_name}_fusion.log")
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
      logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
      logging.StreamHandler(sys.stdout)
    ]
  )
  logger = logging.getLogger()

  logger.info("Starting fusion process.")
  logger.info(f"Model={config.merged_name}")

  token = authenticate(logger)
  create_repo_if_needed(config.merged_repo_id, token,logger)
  model, tokenizer = merge_models(logger)
  save_and_upload(model, tokenizer, token,logger)
  shutil.rmtree(config.merged_output_dir)

if __name__ == "__main__":
  main()