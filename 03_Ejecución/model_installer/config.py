import os

class Config:

#  model_user_id="ndebuhr"
  model_user_id="pedr0gonzalez"
  #model_name = "Mistral-7B-Technical-Tutorial-Summarization-QLoRA"
  #model_name = "MGiver-Mistral-7B-i-qlora-merged-gguf-q4"
  #model_name="Mistral-7B-Technical-Tutorial-Summarization-QLoRA-gguf-q4"
  #model_name="MGiver-2-Mistral-7B-i-qlora-merged-gguf-q4"
  #model_name="MGiver-3-Mistral-7B-i-qlora-merged-gguf-q4"
  model_name="MGiver-4-Mistral-7B-i-qlora-merged-gguf-q4"
  model_filename = f"{model_name}.gguf"
  model_repo_id = f"{model_user_id}/{model_name}"

  model_local_dir = "./models_storage"

  # Logging
  log_dir = "./logs"