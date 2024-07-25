import json
import os
import re
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(verbose=True)
# Create an OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def load_json(filename: str) -> Dict:
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def save_json(data: dict, filename: str) -> None:
    with open(filename, "w") as file:
        json.dump(data, file, indent=2)


def filter_items_without_answer(items: dict, num_items: int) -> List[Dict]:
    filtered_items = [item for item in items if not item["hasanswer"]]
    return filtered_items[:num_items]


def gen_openai_para(
    question: str,
    answer: str,
    num_pairs: int = 9,
    answer_limit: int = 4,
) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"""
                                You are tasked with creating question-answer pairs. You are given a question and its correct answer. 
                                Your job is to create synthetic answers following the guidelines below:

                                1. Answers should be concise and in a closed form.
                                2. Answers should be no longer than {answer_limit} words.
                                3. Avoid sentence form in the answers.
                                4. Do not use the exact same wording as the original answer.
                                5. Provide {num_pairs} synthetic answers for each question and answer pair.

                                Examples:

                                Question: What is the capital of France?
                                Original Answer: Paris
                                Synthetic Answer: Paris, France
                                Synthetic Answer: The city Paris
                                Synthetic Answer: Paris, French capital
                                Synthetic Answer: Capital: Paris
                                Synthetic Answer: Paris city
                                Synthetic Answer: Paris, Europe
                                Synthetic Answer: City of Paris
                                Synthetic Answer: Paris in France
                                Synthetic Answer: Parisian capital

                                Bad Answer: The capital city
                                Bad Answer: France's main city
                                Bad Answer: The French capital

                                Question: What is the largest land animal?
                                Original Answer: African Elephant
                                Synthetic Answer: African elephants
                                Synthetic Answer: Largest elephant
                                Synthetic Answer: African species
                                Synthetic Answer: Elephants, Africa
                                Synthetic Answer: Elephant species
                                Synthetic Answer: African giants
                                Synthetic Answer: Land elephant
                                Synthetic Answer: Elephant, Africa
                                Synthetic Answer: African mammals

                                Bad Answer: largest terrestrial animal
                                Bad Answer: largest land animal
                                Bad Answer: hugest animal

                                Question: When did South Sudan join East African Community?
                                Original Answer: April 2016
                                Synthetic Answer: April, 2016
                                Synthetic Answer: 2016 April
                                Synthetic Answer: In April 2016
                                Synthetic Answer: April '16
                                Synthetic Answer: 04/2016
                                Synthetic Answer: Year 2016
                                Synthetic Answer: Joining: 2016
                                Synthetic Answer: 2016, April
                                Synthetic Answer: 2016 April join

                                Bad Answer: Exactly the same as the Original Answer (April 2016)

                                Question: How long does this event last?
                                Original Answer: April to September
                                Synthetic Answer: Apr-Sep
                                Synthetic Answer: April-Sept
                                Synthetic Answer: From Apr to Sep
                                Synthetic Answer: Apr through Sep
                                Synthetic Answer: Apr until Sep
                                Synthetic Answer: April till Sept
                                Synthetic Answer: April-September
                                Synthetic Answer: April through Sept
                                Synthetic Answer: April until September

                                Bad Answer: Exactly the same as the Original Answer (April to September)
                                """
                    ),
                },
                {"role": "user", "content": f"{question}\n{answer}"},
            ],
            temperature=1.15,
            max_tokens=2328,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"


def gen_openai_counterfactual(
    question: str, answer: str, num_pairs: int = 7, text_limit: int = 30
) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are tasked with creating question-answer pairs. You are given a question and its correct answer. "
                        f"Your job is to create a new, incorrect answer to the given question. For the incorrect answer, write three lines "
                        f"of Wikipedia-style text that assert this incorrect answer as completely true, without mentioning the original correct answer. "
                        f"Generate {num_pairs} incorrect answer-text pairs for each given question-answer pair. Each pair should follow this format: "
                        f"Synthetic Answer: [your synthetic answer]\nSynthetic Text: [text supporting the synthetic answer, answering the original question, limit to {text_limit} words.] "
                        f"Ensure that the text fully supports the synthetic answer as true without referencing the original answer. The explanation should start with directly answering "
                        f"the question followed by a detailed argument, opinion, or claim. Do not imply that the synthetic answer is incorrect in any way."
                    ),
                },
                {"role": "user", "content": f"{question}\n{answer}"},
            ],
            temperature=1.0,
            max_tokens=2328,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"


def parse_synthetic_answer(data: str) -> List[Dict[str, str]]:
    def extract_answers(data: str) -> List[str]:
        if "Synthetic Answer:" in data:
            pattern = re.compile(r"Synthetic Answer:\s*(.+?)(?=\n|$)", re.DOTALL)
            return pattern.findall(data)
        else:
            return [line.strip() for line in data.split("\n") if line.strip()]

    def process_matches(matches: List[str]) -> List[Dict[str, str]]:
        return [{"answer": match} for match in matches]

    matches = extract_answers(data)
    return process_matches(matches)


def parse_synthetic_text(data: str) -> List[Dict[str, str]]:
    def extract_pairs(data: str) -> List[re.Match]:
        pattern = re.compile(
            r"Synthetic Answer:\s*(.+?)\s+Synthetic Text:\s*(.+?)(?=\nSynthetic Answer:|\Z)",
            re.DOTALL,
        )
        return pattern.findall(data)

    def process_matches(matches: List[re.Match]) -> List[Dict[str, str]]:
        return [
            {"answer": match[0].strip(), "text": match[1].strip()} for match in matches
        ]

    matches = extract_pairs(data)
    return process_matches(matches)


def query_and_check(
    question: str,
    answer: str,
    num_pairs: int = 7,
    text_limit=30,
    mode="para",
    answer_limit: int = 5,
) -> List[Dict[str, str]]:
    for _ in range(3):  # Try up to 3 times
        if mode == "para":
            response = gen_openai_para(
                question, answer, num_pairs, answer_limit=answer_limit
            )
        else:
            response = gen_openai_counterfactual(
                question, answer, num_pairs, text_limit
            )
        result = parse_synthetic_answer(response)
        if len(result) == num_pairs:
            return result
    return []


def count_words(input_string: str) -> int:
    words = input_string.split()
    return len(words)


# Example usage
question = "who is the current director of the us mint"
answer = "David J. Ryder"
num_pairs = 9
text_limit = 30
print(count_words(answer))
result = gen_openai_para(
    question=question, answer=answer, num_pairs=num_pairs, answer_limit=4
)
print(result)
result = parse_synthetic_answer(result)
print(result)
