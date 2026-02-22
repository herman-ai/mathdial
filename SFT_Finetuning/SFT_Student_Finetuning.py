import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

####################################################################################################################
# This script fine-tunes Qwen on STUDENT turns from the MathDial dataset.
#
# Key differences from SFT_Finetuning.py (teacher model):
#   - Roles are flipped in the training conversation:
#       teacher (originally "assistant") -> "user"
#       student (originally "user")      -> "assistant"
#   - The system prompt describes the student persona, not the teacher.
#   - Labels are masked on everything EXCEPT student ("assistant") turns.
#   - Ground truth is NOT given to the student (it shouldn't know the answer).
#
# The resulting model can be loaded as QwenStudent in qwen_baseline.py with
# --student_model_path pointing to finetuned_qwen_student_model/.
####################################################################################################################

tokenization_length = 1024

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").to(device)

dataset = load_dataset("eth-nlped/mathdial-chat")


def extract_name(profile: str) -> str:
    return profile.split()[0] if profile else "Student"


def prepare_student_conversations(dataset_split):
    """
    For each example, build a role-flipped conversation:
        [system(student persona)] [user=teacher] [assistant=student] [user=teacher] ...

    For every student ("assistant") turn at position pos, create one training example:
        input  : conversation[:pos]   (context up to but not including this student turn)
        target : conversation[pos]    (the student's actual response)
        labels : -100 for input tokens, real token IDs for target tokens only
    """
    processed_data = []

    for example in tqdm(dataset_split, desc="Processing dataset"):
        raw_conversation = example.get('conversation', [])
        question = example.get('question', '')
        student_incorrect_solution = example.get('student_incorrect_solution', '')
        student_profile = example.get('student_profile', '')
        student_name = extract_name(student_profile)

        if not raw_conversation or len(raw_conversation) < 2:
            continue

        # --- Build student system prompt ---
        # The student knows the problem and their own (incorrect) attempt.
        # They do NOT know the ground truth.
        system_message = {
            "role": "system",
            "content": (
                f"You are {student_name}, a student working through this math problem:\n{question}\n\n"
                f"Your current attempt (which may be wrong) is:\n{student_incorrect_solution}\n\n"
                f"Respond as a real student:\n"
                f"- Speak in first person and show your own reasoning.\n"
                f"- Be uncertain when you are not sure; it is okay to be wrong.\n"
                f"- Keep replies short and natural (2-4 sentences).\n"
                f"- Do not act as a tutor; do not give polished explanations.\n"
                f"- If confused, ask one short clarifying question.\n"
                f"- Make mistakes that are consistent with your incorrect attempt above."
            )
        }

        # --- Flip roles, skip original system and injected incorrect-solution turn ---
        # In eth-nlped/mathdial-chat:
        #   role="assistant" -> teacher/tutor turn
        #   role="user"      -> student turn
        # We skip the first "user" message after the system because the original
        # training code injects the incorrect solution there; we capture that via
        # student_incorrect_solution instead.
        flipped = [system_message]
        skipped_first_user = False

        for msg in raw_conversation:
            role = msg.get('role', '').lower()
            content = msg.get('content', '')

            if role == 'system':
                # Replaced by our student system prompt above
                continue
            if role == 'user' and not skipped_first_user:
                # Skip the injected incorrect-solution message
                skipped_first_user = True
                continue
            if role == 'assistant':
                # Teacher turn -> becomes the "user" prompt seen by the student model
                flipped.append({"role": "user", "content": content})
            elif role == 'user':
                # Student turn -> the target "assistant" output we are training on
                flipped.append({"role": "assistant", "content": content})

        # --- Find student (now "assistant") positions to train on ---
        student_positions = [
            i for i, msg in enumerate(flipped)
            if i > 0 and msg.get('role', '').lower() == 'assistant'
        ]

        if not student_positions:
            continue

        # --- Build one training example per student turn ---
        for pos in student_positions:
            context = flipped[:pos]
            input_text = tokenizer.apply_chat_template(
                context, tokenize=False, add_generation_prompt=True
            )
            full_text = tokenizer.apply_chat_template(
                context + [flipped[pos]], tokenize=False
            )
            encoded_text = tokenizer.encode(full_text, add_special_tokens=False)

            if len(encoded_text) >= tokenization_length:
                # Drop examples that are too long (mirrors original script behavior)
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

            # Mask input/context tokens with -100; only supervise on student response
            len_input = len(
                tokenizer(input_text, add_special_tokens=False)["input_ids"]
            )
            labels = [-100] * len_input + input_ids[len_input:]
            labels = labels[:tokenization_length] + [-100] * (tokenization_length - len(labels))

            processed_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

    return Dataset.from_list(processed_data)


# Prepare datasets
train_dataset_hf = prepare_student_conversations(dataset["train"])
test_dataset_hf = prepare_student_conversations(dataset["test"])

print(f"Training examples : {len(train_dataset_hf)}")
print(f"Evaluation examples: {len(test_dataset_hf)}")

# Training configuration — same hyperparameters as the teacher SFT
training_config = SFTConfig(
    output_dir="./Qwen_SFT_model/finetuned_qwen_student_model",
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

print("Start training student model")
trainer.train()
print("Training complete")

trainer.save_model("./Qwen_SFT_model/finetuned_qwen_student_model")
tokenizer.save_pretrained("./Qwen_SFT_model/finetuned_qwen_student_model")
print("Model saved to ./Qwen_SFT_model/finetuned_qwen_student_model")
