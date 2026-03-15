import argparse
import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--sft_checkpoint", type=str, default="./models/Qwen_SFT_model/finetuned_unweighted_qwen_instruct_teacher_model", help="Path to the SFT checkpoint")
parser.add_argument(
    "--ref_checkpoint",
    type=str,
    default="__SAME_AS_SFT__",
    help="Reference policy checkpoint. Use __SAME_AS_SFT__ to keep old behavior.",
)
parser.add_argument("--output_dir", type=str, default="./models/dpo_qwen_tutor", help="Directory to save checkpoints and final model")
parser.add_argument("--train_data", type=str, default="./data/preference-data/train.jsonl", help="Path to training preference data JSONL")
parser.add_argument("--eval_data", type=str, default="./data/preference-data/eval.jsonl", help="Path to eval preference data JSONL")
parser.add_argument(
    "--force_retrain",
    action="store_true",
    help="If set, train even when output_dir/final already exists",
)
parser.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default="auto",
    help="Checkpoint resume mode: 'auto' (latest checkpoint-* in output_dir), 'none', or explicit checkpoint path",
)
args = parser.parse_args()

final_dir = os.path.join(args.output_dir, "final")
if os.path.isdir(final_dir) and not args.force_retrain:
    print(f"Final checkpoint already exists at {final_dir}; skipping training. Use --force_retrain to override.")
    raise SystemExit(0)


def get_latest_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None
    latest_step = -1
    latest_path = None
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        try:
            step = int(name.split("checkpoint-")[-1])
        except ValueError:
            continue
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and step > latest_step:
            latest_step = step
            latest_path = path
    return latest_path

model_name_or_path = args.sft_checkpoint
ref_name_or_path = args.sft_checkpoint if args.ref_checkpoint in ("", "__SAME_AS_SFT__") else args.ref_checkpoint

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="bfloat16",
    device_map="auto",
)

ref_model = AutoModelForCausalLM.from_pretrained(
    ref_name_or_path,
    torch_dtype="bfloat16",
    device_map="auto",
)

train_dataset = load_dataset(
    "json",
    data_files={"train": args.train_data},
)["train"]

# eval_dataset = load_dataset(
#     "json",
#     data_files={"eval": args.eval_data},
# )["eval"]

training_args = DPOConfig(
    output_dir=args.output_dir,
    # DITTO paper (Appendix C, Table 5): ~24 effective batch size, 40 grad steps
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    gradient_accumulation_steps=8,   # effective batch = 3 * 8 = 24
    learning_rate=1e-6,
    max_steps=40,                    # DITTO: 40 DPO gradient steps per round
    logging_steps=5,
    save_steps=40,                   # save at end of the 40 steps
    bf16=True,
    remove_unused_columns=False,
    max_length=2048,
    max_prompt_length=1536,
    max_completion_length=512,
    beta=0.05,                       # DITTO: beta=0.05
    lr_scheduler_type="constant_with_warmup",  # DITTO: constant_with_warmup
    warmup_ratio=0.25,               # DITTO: warmup_ratio=0.25
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)

resume_arg = None
if args.resume_from_checkpoint.lower() == "auto":
    resume_arg = get_latest_checkpoint(args.output_dir)
elif args.resume_from_checkpoint.lower() == "none":
    resume_arg = None
else:
    resume_arg = args.resume_from_checkpoint

if resume_arg:
    print(f"Resuming from checkpoint: {resume_arg}")
    trainer.train(resume_from_checkpoint=resume_arg)
else:
    print("Starting fresh training run (no checkpoint resume).")
    trainer.train()

trainer.save_model(os.path.join(args.output_dir, "final"))
tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

