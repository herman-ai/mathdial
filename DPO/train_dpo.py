import argparse
import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--sft_checkpoint", type=str, default="./models/Qwen_SFT_model/finetuned_unweighted_qwen_instruct_teacher_model", help="Path to the SFT checkpoint")
parser.add_argument("--output_dir", type=str, default="./models/dpo_qwen_tutor", help="Directory to save checkpoints and final model")
args = parser.parse_args()

model_name_or_path = args.sft_checkpoint

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="bfloat16",
    device_map="auto",
)

ref_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="bfloat16",
    device_map="auto",
)

train_dataset = load_dataset(
    "json",
    data_files={"train": "./data/preference-data/train.jsonl"},
)["train"]

eval_dataset = load_dataset(
    "json",
    data_files={"eval": "./data/preference-data/eval.jsonl"},
)["eval"]

training_args = DPOConfig(
    output_dir=args.output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,   # 8
    learning_rate=5e-7,
    num_train_epochs=1,
    logging_steps=10,  # 10
    eval_steps=100,     # 100
    save_steps=100,     # 100
    bf16=True,
    remove_unused_columns=False,
    max_length=2048,
    max_prompt_length=1536,
    max_completion_length=512,
    beta=0.1,
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
trainer.save_model(os.path.join(args.output_dir, "final"))
tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

