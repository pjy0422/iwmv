from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from numpy import random
from pydantic import BaseModel
from tqdm import tqdm
from utils.json_utils import load_json, save_json
from utils.openai_utils import OpenaiQueryHandler


@dataclass
class YesOrNo(BaseModel):
    Label: str
    Reason: str


def get_system_prompt() -> str:
    return f"""
    I want you to check each question is closed-ended question or not.
    there is question and its answers.
    If a question has correct multi answer, then it is not closed-ended.
    Example: what is the greatest thing?
    But there is tolerance for minor variation, typo and capitalization.
    Who was the next British Prime Minister after Arthur Balfour?
    "Sir Henry Campbell-Bannerman",
    "Henry Campbell-Bannerman",
    "Campbell-Bannerman"
    this is example is closed-ended because all answers are just minor variations.
    there should be closed_ended label and your brief reason.
    label should be only "yes" or "no".
    """


def get_user_prompt(question: str, answers: list) -> str:
    return f"""
    Question: {question}
    Answers: {answers}
    """


def gen_tuple(question: str, answers: list) -> Tuple[str, str, Dict[str, Any]]:
    num_pairs = 5
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(question, answers)
    kwargs = {
        "model": "gpt-4o-mini",
        "max_tokens": 200,
        "top_p": 1,
        "temperature": 0.1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": YesOrNo,
    }
    return system_prompt, user_prompt, kwargs


def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    question = item["question"]
    answers = item["valid_answers"]

    system_prompt, user_prompt, kwargs = gen_tuple(question, answers)

    new_contexts = []
    question_handler = OpenaiQueryHandler(
        system_prompt=system_prompt, user_prompt=user_prompt, **kwargs
    )

    response = question_handler.query_with_schema()

    return {
        "index": item["index"],
        "question": item["question"],
        "answers": item["answers"],
        "answers_in_ctxs": item["valid_answers"],
        "closed_ended": response.Label,
        "reason": response.Reason,
        "target": item["target"],
        "ctxs": item["ctxs"],
    }


def main():
    original_data_path = "data/source/triviaQA_with_validans.json"
    new_data_path = "data/source/triviaQA_labelled_by_mini.json"
    original_data = load_json(original_data_path)
    new_data = []
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {
            executor.submit(process_item, item): item for item in original_data
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing items"
        ):
            try:
                result = future.result()
                new_data.append(result)

            except Exception as e:
                print(f"Error processing item: {e}")
    new_data = sorted(new_data, key=lambda x: x["index"])
    save_json(new_data_path, new_data)


if __name__ == "__main__":
    main()
