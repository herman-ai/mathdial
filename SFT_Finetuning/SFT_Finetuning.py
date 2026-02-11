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
dataset = load_dataset("eth-nlped/mathdial-chat")

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
        conversation = example.get('conversation', [])
        ground_truth = example.get('ground_truth', '')
        student_profile = example.get('student_profile', '')
        student_name = extract_name(student_profile)
        
        if not conversation or len(conversation) < 2:
            continue

        studentprofile = example.get('student_profile', '')
        studentname = extract_name(studentprofile)
        for conv in conversation:
            if isinstance(conv, dict) and 'content' in conv and conv['role'] == 'system':
                conv['content'] = conv['content'].replace(
                    "The student is trying to solve the following problem:",
                    f"The student, with the name {studentname}, is trying to solve the following problem:"
                )

        for conv in conversation:
            if isinstance(conv, dict) and 'role' in conv and conv['role'] == 'system':
                conv['content'] += f"\nThe correct solution is as follows:\n{ground_truth}\n"
                break  # Exit the loop after modifying the first 'system' role
        student_incorrect_solution = example.get('student_incorrect_solution', '')
        student_incorrect_solution_dict = {"content": student_incorrect_solution, "role": "user"}

        for i, conv in enumerate(conversation):
            if conv['role'] == 'system':
                if i + 1 < len(conversation) and conversation[i + 1] == student_incorrect_solution_dict:
                    break
            conversation.insert(i + 1, student_incorrect_solution_dict)
            break 
        assistant_positions = [i for i, msg in enumerate(conversation) if i > 0 and msg.get('role', '').lower() == 'assistant']
        
        if not assistant_positions:
            continue

        for pos in assistant_positions:
            context = conversation[:pos]
            input_text = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(context + [conversation[pos]], tokenize=False)
            encoded_text = tokenizer.encode(full_text, add_special_tokens=False)

            if len(encoded_text) < tokenization_length:
                tokenized = tokenizer(full_text, add_special_tokens=True, truncation=True, padding='max_length', max_length=tokenization_length)
            else:
                continue

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            len_input = len(tokenizer(input_text, add_special_tokens=False)["input_ids"])
            labels = [-100] * len_input + input_ids[len_input:]
            labels = labels[:tokenization_length] + [-100] * (tokenization_length - len(labels))
            processed_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })
    
    return Dataset.from_list(processed_data)

# Prepare training and test datasets
# The model trains only on the "train" dataset, the "test" dataset is used for evaluation during training.
train_dataset_hf = prepare_conversations(dataset["train"])
test_dataset_hf = prepare_conversations(dataset["test"])

# Training configuration
training_config = SFTConfig(
    output_dir=f"./Qwen_SFT_model/finetuned_qwen_model",
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

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=train_dataset_hf,
    eval_dataset=test_dataset_hf
)


# Train model
print("Start training")
trainer.train()
print("Training complete")

# Save fine-tuned model
trainer.save_model(f"./Qwen_SFT_model/finetuned_qwen_model")
tokenizer.save_pretrained(f"./Qwen_SFT_model/finetuned_qwen_model")