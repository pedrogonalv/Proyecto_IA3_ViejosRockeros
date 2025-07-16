import os

class Config:

  source_user_hf_id="pedr0gonzalez"
  #source_user_hf_id="ndebuhr"

  source_model_name="MGiver-4-Mistral-7B-i-qlora-merged"
  #source_model_name="Mistral-7B-Technical-Tutorial-Summarization-QLoRA"

  source_repo_id = f"{source_user_hf_id}/{source_model_name}"

  user_hf_id="pedr0gonzalez"

  gguf_model_name = f"{source_model_name}-gguf"
  gguf_filename = f"{gguf_model_name}.gguf"
  gguf_repo_id = f"{user_hf_id}/{gguf_model_name}"
  quantized_model_name = f"{gguf_model_name}-q4"
  quantized_filename = f"{quantized_model_name}.gguf"
  quantized_repo_id = f"{user_hf_id}/{quantized_model_name}"

  source_local_dir = f"./{source_model_name}"
  gguf_output_dir = f"./{gguf_model_name}"
  quantized_output_dir = f"./{quantized_model_name}"


  llama_cpp_dir = "../llama.cpp"
  convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
  llama_cpp_bin = os.path.join(llama_cpp_dir, "build/bin/llama-quantize")

  quantize_type = "q4_0"
  gguf_outtype = "f16"

  override_chat_template = True
  chat_template_filename = "react_minimal.jinja"

  upload_gguf = False               # Subir modelo GGUF a HuggingFace
  upload_quantized = True         # Subir modelo cuantizado a HuggingFace
  install_in_ollama = False        # Instalar modelo cuantizado en Ollama local

  log_dir = "./logs"