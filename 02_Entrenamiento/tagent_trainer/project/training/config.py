import os

class Config:

  source_model_name="Mistral-7B-Instruct-v0.3"
  source_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

#  source_model_name="Nous-Hermes-2-Mistral-7B-DPO"
#  source_repo_id = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"

  adapter_name="MGiver-4-Mistral-7B-i-qlora"
  adapter_repo_ser="pedr0gonzalez"
  adapter_output_path=f"../{adapter_name}"
  adapter_repo_id=f"{adapter_repo_ser}/{adapter_name}"
  adapter_repo_private=False

  merged_name=f"{adapter_name}-merged"
  merged_output_dir=f"../{merged_name}"
  merged_repo_id = f"{adapter_repo_ser}/{merged_name}"
  merged_repo_private = False

  source_training_data_path= "../data/mgiver_data.jsonl"
  processed_training_data_path= "../data/mgiver_data_processed.jsonl"
  processed_data_react_ratio=0.4

  log_dir="./log"
