import os
import sys
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, EarlyStoppingCallback, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from huggingface_hub import login, create_repo, upload_folder
from getpass import getpass
import torch
from config import Config
import logging

config=Config()
log_dir = config.log_dir
os.makedirs(log_dir, exist_ok=True)

now = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(config.log_dir, f"{now}_{config.adapter_name}_fine-tuning.log")
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s %(levelname)s: %(message)s",
  handlers=[
    logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
    logging.StreamHandler(sys.stdout)
  ]
)
logger = logging.getLogger()

logger.info("Starting training process.")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/workspace/hugging_face_res"


logger.info("Authentication with Huggingface Hub.")
print("Insert Hugging Face Token:")
hf_token = getpass("Token HF: ")
login(hf_token)

logger.info("Creation of adapter repo.")
try:
  create_repo(config.adapter_repo_id, private=config.adapter_repo_private)
except Exception:
  logger.warning("Repo already exists or couldn't be created.")

logger.info("Loading dataset.")
dataset = load_dataset("json", data_files=config.processed_training_data_path)["train"]
split_dataset = dataset.train_test_split(test_size=0.1, seed=0)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

logger.info("Creating tokenizer.")
tokenizer = AutoTokenizer.from_pretrained(config.source_repo_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_compute_dtype=torch.float16
)
logger.info("Loading source model.")
model = AutoModelForCausalLM.from_pretrained(
  config.source_repo_id,
  quantization_config=bnb_config,
  device_map="cuda:0"
)

lora_config = LoraConfig(
  r=64,
  lora_alpha=16,
  target_modules=["q_proj", "v_proj"],
  lora_dropout=0.05,
  bias="none",
  task_type=TaskType.CAUSAL_LM
)

logger.info("Setting lora configuration.")
model = get_peft_model(model, lora_config)

def preprocess(sample):
  user_input=sample["instruction"].strip()
  model_output=sample["output"].strip()
  is_react=sample.get("react",False)

  if is_react:
    prompt=(
      f"User: {user_input}\n"
      f"Assistant:\n{model_output}"
    )
  else:
    doc_name=sample.get("doc_name","").strip()
    doc_page=sample.get("doc_page","").strip()
    page_info=f"on page {doc_page}" if doc_page not in ["",None,"Unknown"] else "somewhere in the document"

    prompt=(
      f"Context:\n{sample.get('context','').strip()}\n\n"
      f"Source: {doc_name}, {page_info}\n\n"
      f"User: {user_input}\n"
      f"Assistant:\n{model_output}"
    )

  inputs=tokenizer(prompt,truncation=True,padding="max_length",max_length=512)
  inputs["labels"]=inputs["input_ids"].copy()
  return inputs

tokenized_train = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(preprocess, remove_columns=val_dataset.column_names)

def get_training_params(train_size):
  if train_size < 10000:
    batch_size = 6
    grad_accum = 4
    epochs = 8
  elif train_size < 25000:
    batch_size = 4   # 🔽 bajamos de 16 a 4
    grad_accum = 8   # 🔼 compensa con acumulación
    epochs = 6
  elif train_size < 50000:
    batch_size = 4
    grad_accum = 12
    epochs = 5
  else:
    batch_size = 4
    grad_accum = 16
    epochs = 4

  return batch_size, grad_accum, epochs
logger.info("Setting training parameters.")
train_size = len(tokenized_train)
batch_size, gradient_accumulation_steps, num_train_epochs = get_training_params(train_size)

steps_per_epoch = max(1, train_size // (batch_size * gradient_accumulation_steps))

logging_steps = max(10, steps_per_epoch // 10)
eval_steps = max(30, steps_per_epoch // 5)
multiplier = max(1, (steps_per_epoch // 2) // eval_steps)
save_steps = eval_steps * multiplier

training_args = TrainingArguments(
  output_dir=config.adapter_output_path,
  per_device_train_batch_size=batch_size,
  gradient_accumulation_steps=gradient_accumulation_steps,
  num_train_epochs=num_train_epochs,
  learning_rate=2e-4,
  bf16=True,
  logging_steps=logging_steps,
  evaluation_strategy="steps",
  eval_steps=eval_steps,
  save_strategy="steps",
  save_steps=save_steps,
  save_total_limit=2,
  load_best_model_at_end=True,
  metric_for_best_model="eval_loss",
  greater_is_better=False,
  report_to="none",
  dataloader_num_workers=8  # Use CPU resources for faster data loading
)

logger.info("Setting up trainer")
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_train,
  eval_dataset=tokenized_val,
  callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

logger.info("Train start.")
train_output=trainer.train()
logger.info(train_output)
logger.info("Saving training result")
model.save_pretrained(config.adapter_output_path)
tokenizer.save_pretrained(config.adapter_output_path)

logger.info("Uploading model")
upload_folder(
  repo_id=config.adapter_repo_id,
  folder_path=config.adapter_output_path,
  commit_message="Upload fine-tuned QLoRA adapter",
  token=hf_token
)

logger.info("Process finished.")