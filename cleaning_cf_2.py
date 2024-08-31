from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from numpy import random
from pydantic import BaseModel
from tqdm import tqdm
from utils.dataclass_utils import CF_Cleaning
from utils.json_utils import load_json, save_json
from utils.openai_utils import OpenaiQueryHandler


def get_system_prompt() -> str:
    return f"""
You are given a question, an answer, text.
Put the given answer to the text, do not use any paraphrasing.
Double check the given answer is in the text.
"""


def get_user_prompt(question: str, answer: str, text: str) -> str:
    text1 = text
    return f"""
    Question: {question}
    Answer: {answer}
    text1: {text1}
    """


def gen_tuple(
    question: str, answer: str, text: list
) -> Tuple[str, str, Dict[str, Any]]:
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(question, answer, text)
    kwargs = {
        "model": "gpt-4o-mini",
        "max_tokens": 1200,
        "top_p": 1,
        "temperature": 0.9,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": CF_Cleaning,
    }
    return system_prompt, user_prompt, kwargs


def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    question = item["question"]
    tuple_list = []
    cf_list = []

    # Create a list of tuples with system_prompt, user_prompt, kwargs, and answer
    for cf in item["counterfactual"]:
        answer = cf["answers"][0]
        text = cf["contexts"]
        if [answer.lower() in t.lower() for t in text] == [True, True, True]:
            cf_list.append(cf)
        for t in text:
            if answer.lower() not in t.lower():
                system_prompt, user_prompt, kwargs = gen_tuple(
                    question, answer, t
                )
                tuple_list.append((system_prompt, user_prompt, kwargs, answer))
                break
    # Create handlers and associate them with their corresponding answers
    question_handler_list = [
        (
            OpenaiQueryHandler(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=CF_Cleaning,
                kwargs=kwargs,
            ),
            answer,
        )
        for system_prompt, user_prompt, kwargs, answer in tuple_list
    ]

    with ThreadPoolExecutor(max_workers=9) as executor:
        for handler, answer in question_handler_list:
            results = None
            while results is None or len(results.texts) != 3:
                future = executor.submit(handler.query_with_schema)
                results = future.result()
                if len(results.texts) == 3:
                    cf_list.append(
                        {
                            "answers": [answer],
                            "contexts": results.texts,
                        }
                    )

    return {
        "index": item["index"],
        "question": item["question"],
        "answers": item["answers"],
        "paraphrase": item["paraphrase"],
        "counterfactual": cf_list,
    }


def main():
    original_data_path = (
        "/home/guest-pjy/data/0830/hotpotqa_postprocessed.json"
    )
    new_data_path = "/home/guest-pjy/data/0831/hotpot_cf_cleaned.json"
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

    for idx, item in enumerate(new_data):
        item["index"] = idx
    for item in new_data:
        if len(item["counterfactual"]) != 9:
            print(item["index"])
        for cf in item["counterfactual"]:
            if len(cf["contexts"]) != 3:
                print(item["index"])
    save_json(new_data_path, new_data)


if __name__ == "__main__":
    main()
