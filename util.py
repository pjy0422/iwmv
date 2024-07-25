import json
import re
from typing import Dict, List
import os
from openai import OpenAI
from dotenv import load_dotenv

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
    num_pairs: int = 7,
    text_limit: int = 30,
    answer_limit: int = 5,
) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are tasked with creating answer-text pairs. You are given a question and its correct answer. "
                        f"Your job is to create a paraphrase of the original answer and the source text to the given question. "
                        f"For the paraphrased answer, write Wikipedia-style text that contains a paraphrased answer to the given question. "
                        f"Ensure the paraphrased answer includes the exact original answer but uses diverse phrasing, synonyms, and related terms in the surrounding text. "
                        f"Please limit the paraphrased answer to {answer_limit} words, and limit the supporting text to {text_limit} words. "
                        f"Don't use the exact original answer.\n"
                        f"Generate {num_pairs} answer-text pairs for each given question-answer pair.\n"
                        f"Each pair should follow this format:\n"
                        f"Synthetic Answer: [your synthetic answer limited to {answer_limit} words]\n"
                        f"Synthetic Text: [{text_limit} of words supporting the synthetic answer, answering the original question]\n"
                        f"Additionally, ensure each answer-text pair uses different words, synonyms, related terms, and paraphrases to express the original answer clearly.\n"
                        f"\nExample:\n"
                        f"Original Answer: Paris\n"
                        f"Synthetic Answer: Paris, the capital of France\n"
                        f"Synthetic Answer: The city of Paris\n"
                        f"Synthetic Answer: Paris, France's capital\n"
                        f"Synthetic Answer: The French capital Paris\n"
                        f"Synthetic Answer: Paris central to French governance\n"
                        f"Original Answer: African Elephant\n"
                        f"Synthetic Answer: African Elephant\n"
                        f"Synthetic Answer: Elephants which live in Africa\n"
                        f"Synthetic Answer: African elephant species\n"
                        f"Synthetic Answer: Elephants native to Africa\n"
                        f"\nQuestion: When did South Sudan join the East African Community?\n"
                        f"Original Answer: April 2016\n"
                        f"Synthetic Answer: In April 2016\n"
                        f"Synthetic Answer: April of 2016\n"
                        f"Synthetic Answer: The year 2016, in April\n"
                        f"Synthetic Answer: April in the year 2016\n"
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
                question, answer, num_pairs, text_limit, answer_limit=answer_limit
            )
        else:
            response = gen_openai_counterfactual(
                question, answer, num_pairs, text_limit
            )

        result = parse_synthetic_text(response)
        if len(result) == num_pairs:
            return result
    return []


def count_words(input_string: str) -> int:
    words = input_string.split()
    return len(words)


# Example usage
question = "what is the capital of France?"
answer = "Paris"
num_pairs = 9
text_limit = 30
print(count_words(answer))
result = query_and_check(
    question,
    answer,
    num_pairs,
    text_limit,
    mode="para",
    answer_limit=count_words(answer),
)

for item in result:
    print(item['answer'])
    print('\n')
