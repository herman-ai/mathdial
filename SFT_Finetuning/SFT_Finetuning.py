import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm
####################################################################################################################
#This code takes the pretrained Qwen model and fine-tunes it on the MathDial dataset. 
#The students Ground truth is added to the system prompt.
#and the students incorrect solution is added to the conversation.
####################################################################################################################
tokenization_length = 1024

# Load model and tokenizer. You could also use a different model here.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").to(device)

# Load MathDial dataset
# dataset = load_dataset("eth-nlped/mathdial-chat")
dataset = load_dataset("eth-nlped/mathdial")

#Extract student name from profile
def extract_name(profile: str) -> str:
    return profile.split()[0] if profile else "Student"


#The following function prepares the conversations by adding the ground truth to the system prompt
#and the students incorrect solution to the conversation.
#Then it builds the conversation like this:
#Input: Systemprompt, Student incorrect solution. Output: Assistant
#Nextinput: Systemprompt, Student incorrect solution, Assistant, Students response Output: Assistant response
#and so forth. Then it applies the chat template + tokenization.
def prepare_conversations(dataset_split):
    processed_data = []
    for example in tqdm(dataset_split, desc="Processing dataset"):
        raw_conversation = example.get('conversation', '')
        ground_truth = example.get('ground_truth', '')
        question = example.get('question', '')
        student_profile = example.get('student_profile', '')
        student_incorrect_solution = example.get('student_incorrect_solution', '')
        student_name = extract_name(student_profile)

        if not raw_conversation:
            continue

        # --- Build teacher system prompt (teacher knows the ground truth) ---
        system_message = {
            "role": "system",
            "content": (
                f"You are a math tutor helping {student_name} solve the following problem:\n{question}\n\n"
                f"The correct solution is as follows:\n{ground_truth}\n\n"
                f"The student's current (incorrect) attempt is:\n{student_incorrect_solution}"
            )
        }

        # --- Parse "|EOM|"-delimited turns from the mathdial string format ---
        # Teacher turns -> "assistant" (the target output we train on)
        # Student turns -> "user" (the input context)
        turns = [t.strip() for t in raw_conversation.split('|EOM|') if t.strip()]
        if len(turns) < 2:
            continue

        # Prepend the student's incorrect solution as the first user message so the
        # teacher has context before the first real student turn.
        conversation = [
            system_message,
            {"role": "user", "content": student_incorrect_solution},
        ]
        for turn in turns:
            if ': ' not in turn:
                continue
            speaker, content = turn.split(': ', 1)
            speaker = speaker.strip()
            content = content.strip()
            if speaker.lower() == 'teacher':
                conversation.append({"role": "assistant", "content": content})
            else:
                # Any non-teacher speaker is the student
                conversation.append({"role": "user", "content": content})

        # --- Find teacher ("assistant") positions to train on ---
        assistant_positions = [
            i for i, msg in enumerate(conversation)
            if i > 0 and msg.get('role', '').lower() == 'assistant'
        ]

        if not assistant_positions:
            continue

        for pos in assistant_positions:
            context = conversation[:pos]
            input_text = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(context + [conversation[pos]], tokenize=False)
            encoded_text = tokenizer.encode(full_text, add_special_tokens=False)

            if len(encoded_text) >= tokenization_length:
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
            len_input = len(tokenizer(input_text, add_special_tokens=False)["input_ids"])
            labels = [-100] * len_input + input_ids[len_input:]
            labels = labels[:tokenization_length] + [-100] * (tokenization_length - len(labels))
            processed_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "weight": (example.get('self-typical-confusion') or 0) + (example.get('self-typical-interactions') or 0)
            })

    return Dataset.from_list(processed_data)

# Prepare training and test datasets
# The model trains only on the "train" dataset, the "test" dataset is used for evaluation during training.
train_dataset_hf = prepare_conversations(dataset["train"])
test_dataset_hf = prepare_conversations(dataset["test"])

# Training configuration
training_config = SFTConfig(
    output_dir=f"./models/Qwen_SFT_model/finetuned_weighted_qwen_instruct_teacher_model",
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

##################################################################
# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=train_dataset_hf,
    eval_dataset=test_dataset_hf
)

def compute_weights(dataset):
    scores = torch.tensor([example['weight'] for example in dataset], dtype=torch.float)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores

def custom_dataloader_function(self):
    weights = compute_weights(self.train_dataset)
    train_dataset = self.train_dataset.remove_columns(["weight"])
    print("Weights computed for training samples.")
    print(f"Sample weights: {weights[:10]}")  # Print first 10 weights for verification
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return torch.utils.data.DataLoader(train_dataset, 
                                       batch_size=self.args.per_device_train_batch_size, 
                                       sampler=sampler, 
                                       collate_fn=self.data_collator, 
                                       num_workers=self.args.dataloader_num_workers,
                                       pin_memory=self.args.dataloader_pin_memory)

trainer.get_train_dataloader = custom_dataloader_function.__get__(trainer)
##################################################################

# Train model
print("Start training")
trainer.train()
print("Training complete")

# Save fine-tuned model
trainer.save_model(f"./models/Qwen_SFT_model/finetuned_weighted_qwen_instruct_teacher_model")
tokenizer.save_pretrained(f"./models/Qwen_SFT_model/finetuned_weighted_qwen_instruct_teacher_model")