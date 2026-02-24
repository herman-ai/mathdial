from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

####################################################################################################################
# Fine-tunes Qwen2.5-1.5B BASE (no Instruct) on TEACHER turns from MathDial.
#
# Key differences from SFT_Finetuning.py (instruct teacher):
#   - Base model: no chat template, no RLHF alignment.
#   - Training format is raw text completion, not chat messages.
#   - Prompt format matches BaseModelTeacher.PROMPT_TEMPLATE in qwen_base_teacher.py
#     exactly, so inference conditions match training conditions.
#
# For each teacher turn in a conversation the training example is:
#   Input (masked, -100):
#       "The following is a math tutoring session.\n
#        The teacher knows the correct solution to the problem.\n\n
#        Problem: {question}\n
#        Correct solution: {ground_truth}\n\n
#        Teacher: {turn1}\nStudent: {turn2}\n...Teacher:"
#   Target (loss computed here):
#       "{teacher response text}\n"
####################################################################################################################

tokenization_length = 1024

PROMPT_HEADER = (
    "The following is a math tutoring session.\n"
    "The teacher knows the correct solution to the problem.\n\n"
    "Problem: {question}\n"
    "Correct solution: {ground_truth}\n\n"
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")

dataset = load_dataset("eth-nlped/mathdial-chat")


def prepare_base_teacher_conversations(dataset_split):
    """
    Build raw-text completion training examples for each teacher turn.

    For each assistant (teacher) turn at position pos, produces:
        full_text  = prompt_up_to_teacher_colon + teacher_response + "\n"
        labels     = [-100] * len(prompt_tokens) + target_token_ids

    The prompt format mirrors BaseModelTeacher.PROMPT_TEMPLATE so that
    the model sees identical context at inference and training time.
    """
    processed_data = []

    for example in tqdm(dataset_split, desc="Processing dataset"):
        raw_conversation = example.get('conversation', [])
        question = example.get('question', '')
        ground_truth = example.get('ground_truth', '')

        if not raw_conversation or len(raw_conversation) < 2:
            continue

        # Build a flat list of (speaker, text) tuples, skipping system and the
        # injected incorrect-solution user turn (first user message after system).
        # In mathdial-chat: role="assistant" -> teacher, role="user" -> student.
        turns = []          # list of ("Teacher"|"Student", text)
        skipped_first_user = False

        for msg in raw_conversation:
            role = msg.get('role', '').lower()
            content = msg.get('content', '').strip()
            if role == 'system':
                continue
            if role == 'user' and not skipped_first_user:
                skipped_first_user = True   # skip injected incorrect solution
                continue
            if role == 'assistant':
                turns.append(("Teacher", content))
            elif role == 'user':
                turns.append(("Student", content))

        if not turns:
            continue

        header = PROMPT_HEADER.format(question=question, ground_truth=ground_truth)

        # For each teacher turn, build one training example
        history_lines = []
        for speaker, text in turns:
            if speaker == "Teacher":
                # Prompt: header + dialogue so far + "Teacher:"
                prompt = header + "".join(history_lines) + "Teacher:"
                # Full text: prompt + space + teacher response + newline
                full_text = prompt + " " + text + "\n"

                encoded_full = tokenizer.encode(full_text, add_special_tokens=False)
                if len(encoded_full) >= tokenization_length:
                    # Update history and skip — example too long
                    history_lines.append(f"Teacher: {text}\n")
                    continue

                tokenized = tokenizer(
                    full_text,
                    add_special_tokens=True,
                    truncation=True,
                    padding='max_length',
                    max_length=tokenization_length
                )
                input_ids = tokenized["input_ids"]
                attention_mask = tokenized["attention_mask"]

                # Mask the prompt. full_text is tokenized with add_special_tokens=True,
                # which prepends a BOS token at index 0. Prompt tokens therefore occupy
                # indices 1..len_prompt, so we must mask len_prompt+1 positions.
                len_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) + 1
                labels = [-100] * len_prompt + input_ids[len_prompt:]
                labels = labels[:tokenization_length]
                # Mask padding positions (pad_token == eos_token, distinguish via attention_mask)
                labels = [-100 if attention_mask[i] == 0 else labels[i] for i in range(tokenization_length)]

                processed_data.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                })

                # Append this turn to history for next iteration
                history_lines.append(f"Teacher: {text}\n")

            elif speaker == "Student":
                history_lines.append(f"Student: {text}\n")

    return Dataset.from_list(processed_data)


# Prepare datasets
train_dataset_hf = prepare_base_teacher_conversations(dataset["train"])
test_dataset_hf = prepare_base_teacher_conversations(dataset["test"])

print(f"Training examples : {len(train_dataset_hf)}")
print(f"Evaluation examples: {len(test_dataset_hf)}")

# Training configuration — same hyperparameters as the other SFT scripts
training_config = SFTConfig(
    output_dir="./models/Qwen_SFT_model/finetuned_qwen_base_teacher_model",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=2000,
    eval_strategy="steps",
    eval_steps=2000,
    optim="adamw_torch",
    learning_rate=6.25e-5,
    weight_decay=0.01,
    fp16=False,
    max_length=tokenization_length
)

trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=train_dataset_hf,
    eval_dataset=test_dataset_hf
)

print("Start training base teacher model")
trainer.train()
print("Training complete")

trainer.save_model("./models/Qwen_SFT_model/finetuned_qwen_base_teacher_model")
# Remove chat template before saving — this is a base completion model, not a chat model.
# qwen_base_teacher.py uses raw text prompts, not apply_chat_template.
tokenizer.chat_template = None
tokenizer.save_pretrained("./models/Qwen_SFT_model/finetuned_qwen_base_teacher_model")
print("Model saved to ./models/Qwen_SFT_model/finetuned_qwen_base_teacher_model")
