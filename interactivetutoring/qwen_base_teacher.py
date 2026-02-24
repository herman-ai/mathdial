import argparse
import json
import re
import torch
import sys

# Workaround for torchvision version conflicts in containers
sys.path = [p for p in sys.path if 'dist-packages' not in p] + [p for p in sys.path if 'dist-packages' in p]

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from history import History
from message import Message
from roles import Roles
from utils import read_jsonl


class BaseModelTeacher:
    """
    Teacher that uses a raw base model (no instruction tuning) via text completion.

    Instead of chat templates, the full conversation is formatted as a plain-text
    dialogue block ending with "Teacher:" and the model is asked to continue it.
    This gives the true zero-shot baseline before any alignment/fine-tuning.
    """

    PROMPT_TEMPLATE = (
        "The following is a math tutoring session.\n"
        "The teacher knows the correct solution to the problem.\n\n"
        "Problem: {question}\n"
        "Correct solution: {ground_truth}\n\n"
        "{history}"
        "Teacher:"
    )

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def reset(self):
        pass

    def _build_history(self, history: History) -> str:
        lines = []
        for msg in history.messages:
            if msg.persona == Roles.TEACHER:
                lines.append(f"Teacher: {msg.text}")
            elif msg.persona == Roles.STUDENT:
                lines.append(f"Student: {msg.text}")
        return "\n".join(lines) + "\n" if lines else ""

    def _stop_at_turn_boundary(self, text: str) -> str:
        """Trim generated text at the next speaker boundary."""
        # Stop if the model starts a new Student or Teacher turn
        match = re.search(r'\n(Teacher|Student)\s*:', text)
        if match:
            text = text[:match.start()]
        return text.strip()

    def response(self, history: History, question: str, ground_truth_solution: str) -> str:
        prompt = self.PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth_solution,
            history=self._build_history(history),
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        )
        return self._stop_at_turn_boundary(response)


class SFTStudent:
    """
    Student that uses the SFT fine-tuned student model via chat template.
    Identical to QwenStudent in qwen_baseline.py.
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.name = "Alex"

    def reset(self):
        pass

    def response(self, history: History, question: str, incorrect_solution: str) -> str:
        conversation = [
            {
                "role": "system",
                "content": (
                    f"You are {self.name}, a real student working through this math problem: {question}\n"
                    f"Your current (possibly wrong) attempt is: {incorrect_solution}\n\n"
                    f"Student behavior policy:\n"
                    f"- Speak in first person as a learner, not a tutor.\n"
                    f"- Be uncertain sometimes and show partial understanding.\n"
                    f"- Give short, natural reasoning (2-5 sentences).\n"
                    f"- Do not present a polished final solution unless asked directly.\n"
                    f"- If confused, ask a brief clarifying question.\n"
                    f"- Keep mistakes realistic and consistent with your incorrect attempt."
                )
            }
        ]

        for msg in history.messages:
            if msg.persona == Roles.TEACHER:
                conversation.append({"role": "user", "content": msg.text})
            elif msg.persona == Roles.STUDENT:
                conversation.append({"role": "assistant", "content": msg.text})

        text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.95,
                top_p=0.95,
                top_k=60,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        )
        return response.strip()


def get_args():
    parser = argparse.ArgumentParser(
        description="Tutoring simulation using a base (non-instruct) Qwen teacher and SFT student."
    )
    parser.add_argument("--input_file", type=str, default="data/test.jsonl")
    parser.add_argument("--export_file", type=str, default="output/qwen_base_teacher_output.jsonl")
    parser.add_argument("--model_name", type=str, default="qwen_base_teacher",
                        help="Key used to store conversations in the output JSONL.")
    parser.add_argument("--teacher_model_path", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="Base (non-instruct) model to use as teacher.")
    parser.add_argument("--student_model_path", type=str,
                        default="Qwen_SFT_model/finetuned_qwen_student_model",
                        help="SFT fine-tuned student model path.")
    parser.add_argument("--max_utterances", type=int, default=4)
    return parser.parse_args()


def export_to_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for conversation in data:
            f.write(json.dumps(conversation) + '\n')


def print_conversation(question, ground_truth_solution, incorrect_solution, history):
    print("\n\n## Conversation")
    print(f"Question: {question}")
    print(f"Correct solution: {ground_truth_solution}")
    print(f"Incorrect solution: {incorrect_solution}")
    print(history)


if __name__ == '__main__':
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading teacher model (base): {args.teacher_model_path}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
    # Ensure pad token is set for base models (they often lack it)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path).to(device)
    print(f"Teacher model loaded on {device}")

    print(f"Loading student model (SFT): {args.student_model_path}")
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_path)
    student_model = AutoModelForCausalLM.from_pretrained(args.student_model_path).to(device)
    print(f"Student model loaded on {device}")

    conversations = []
    data = read_jsonl(args.input_file)

    teacher = BaseModelTeacher(teacher_model, teacher_tokenizer, device)
    student = SFTStudent(student_model, student_tokenizer, device)

    for problem in tqdm(data):
        question = problem["question"]
        ground_truth_solution = problem["ground_truth"]
        incorrect_solution = problem["student_incorrect_solution"]

        history = History()
        teacher.reset()
        student.reset()
        history.add_message(Message(Roles.TEACHER, f"Hi {student.name}! Could you walk me through your solution?"))

        for i in range(args.max_utterances):
            student_message = Message(Roles.STUDENT, student.response(history, question, incorrect_solution))
            history.add_message(student_message)

            teacher_response_message = Message(Roles.TEACHER, teacher.response(history, question, ground_truth_solution))
            history.add_message(teacher_response_message)

        problem[args.model_name] = history.to_delimited_string("<EOM>")
        conversations.append(problem)

        print_conversation(question, ground_truth_solution, incorrect_solution, history)

    export_to_jsonl(conversations, args.export_file)
    print(f"\nExported {len(conversations)} conversations to {args.export_file}")
