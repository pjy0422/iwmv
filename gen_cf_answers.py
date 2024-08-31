from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from numpy import random
from tqdm import tqdm
from utils.dataclass_utils import CF_Answers
from utils.json_utils import load_json, save_json
from utils.openai_utils import OpenaiQueryHandler


def get_system_prompt() -> str:
    return f"""Generate nine counterfactual answers, based on the question and its original answers. 
    Ensure that each counterfactual answer is a plausible but incorrect response, clearly different from the original answers.
    Avoid repeating or paraphrasing the original answer or question.
    The counterfactual answers should be relevant to the context but should introduce a distinct and clearly incorrect or alternative response.
    You should write the answers in short closed form, limit to maximum 4 words length.
    The answers should not be sentence form, but rather a short phrase or word.
    Write in English."""


def get_user_prompt(question: str, answer_list: list) -> str:
    return f"""
        Question: {question}
        Answers: {','.join(answer_list)}
        """


def gen_tuple(question: str, answer: str) -> Tuple[str, str, Dict[str, Any]]:
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(question, answer)
    kwargs = {
        "model": "gpt-4o-mini",
        "max_tokens": 256,
        "top_p": 1,
        "temperature": 0.9,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": CF_Answers,
    }
    return system_prompt, user_prompt, kwargs


def process_cf_answer(item: Dict[str, Any]) -> Dict[str, Any]:

    question = item["question"]
    valid_answers_list = item["answers"]
    system_prompt, user_prompt, kwargs = gen_tuple(
        question, valid_answers_list
    )
    item["counterfactual_answers"] = []
    while len(item["counterfactual_answers"]) != 9:
        question_handler = OpenaiQueryHandler(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=CF_Answers,
            kwargs=kwargs,
        )
        results = question_handler.query_with_schema()
        item["counterfactual_answers"] = results.answers
    return {
        "index": item["index"],
        "question": item["question"],
        "answers": item["answers"],
        "counterfactual_answers": item["counterfactual_answers"],
        "ctxs": item["ctxs"],
    }


def main():
    original_data_path = (
        "/home/guest-pjy/data/0830/hotpot_easy_only_preprocessed.json"
    )
    new_data_path = "/home/guest-pjy/data/0830/hotpot_cf_answers.json"
    original_data = load_json(original_data_path)
    new_data = []
    with ThreadPoolExecutor(max_workers=256) as executor:
        futures = {
            executor.submit(process_cf_answer, item): item
            for item in original_data
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing items"
        ):
            try:
                result = future.result()
                new_data.append(result)

            except Exception as e:
                print(f"Error processing item: {e}")
    for idx, item in enumerate(new_data):
        item["index"] = idx
    for item in new_data:
        if len(item["counterfactual_answers"]) != 9:
            print(item["index"])
    save_json(new_data_path, new_data)


if __name__ == "__main__":
    main()
