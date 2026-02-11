import argparse
import json
import torch
import sys

# Workaround for torchvision version conflicts in containers
# Remove system package paths that might have incompatible torchvision
sys.path = [p for p in sys.path if 'dist-packages' not in p] + [p for p in sys.path if 'dist-packages' in p]

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from history import History
from message import Message
from roles import Roles
from utils import read_jsonl


class QwenStudent:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.name = "Alex"
    
    def reset(self):
        pass
    
    def response(self, history: History, question: str, incorrect_solution: str) -> str:
        """Generate student response using Qwen model"""
        conversation = [
            {
                "role": "system",
                "content": f"You are {self.name}, a student working through a math problem. "
                           f"You have attempted this solution: {incorrect_solution}. "
                           f"Respond naturally as a student would, explaining your thinking."
            }
        ]
        
        # Add conversation history
        for msg in history.messages:
            if msg.persona == Roles.TEACHER:
                conversation.append({"role": "user", "content": msg.text})
            elif msg.persona == Roles.STUDENT:
                conversation.append({"role": "assistant", "content": msg.text})
        
        # Generate response
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return response.strip()


class QwenTeacher:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def reset(self):
        pass
    
    def response(self, history: History, question: str, ground_truth_solution: str) -> str:
        """Generate teacher response using Qwen model"""
        conversation = [
            {
                "role": "system",
                "content": f"You are a helpful math tutor. Guide the student through solving this problem: {question}\n"
                           f"The correct solution is: {ground_truth_solution}\n"
                           f"Help the student understand their mistakes and guide them toward the correct solution."
            }
        ]
        
        # Add conversation history
        for msg in history.messages:
            if msg.persona == Roles.TEACHER:
                conversation.append({"role": "assistant", "content": msg.text})
            elif msg.persona == Roles.STUDENT:
                conversation.append({"role": "user", "content": msg.text})
        
        # Generate response
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return response.strip()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/example.jsonl")
    parser.add_argument("--export_file", type=str, default="output/qwen_model_output.jsonl")
    parser.add_argument("--model_name", type=str, default="qwen_baseline")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Path to Qwen model or HuggingFace model ID")
    parser.add_argument("--max_utterances", type=int, default=4)
    return parser.parse_args()


def export_to_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as output_file:
        for conversation in data:
            output_file.write(json.dumps(conversation) + '\n')


def print_conversation(question: str, ground_truth_solution: str, incorrect_solution: str, history: History):
    print("\n\n## Conversation")
    print(f"Question: {question}")
    print(f"Correct solution: {ground_truth_solution}")
    print(f"Incorrect solution: {incorrect_solution}")
    print(history)


if __name__ == '__main__':
    args = get_args()
    
    # Load Qwen model and tokenizer
    print(f"Loading model: {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    print(f"Model loaded on {device}")
    
    conversations = []
    data = read_jsonl(args.input_file)
    
    student = QwenStudent(model, tokenizer, device)
    teacher = QwenTeacher(model, tokenizer, device)

    for problem in tqdm(data):
        question = problem["question"]
        ground_truth_solution = problem["ground_truth"]
        incorrect_solution = problem["student_incorrect_solution"]

        history = History()
        student.reset()
        teacher.reset()
        history.add_message(Message(Roles.TEACHER, "Hi " + student.name + "! Could you walk me through your solution?"))

        for i in range(args.max_utterances):
            student_message = Message(Roles.STUDENT, student.response(history, question, incorrect_solution))
            history.add_message(student_message)

            teacher_response_message = Message(Roles.TEACHER,
                                               teacher.response(history, question, ground_truth_solution))
            history.add_message(teacher_response_message)

        problem[args.model_name] = history.to_delimited_string("<EOM>")
        conversations.append(problem)

        print_conversation(question, ground_truth_solution, incorrect_solution, history)

    export_to_jsonl(conversations, args.export_file)
    print(f"\nExported {len(conversations)} conversations to {args.export_file}")
