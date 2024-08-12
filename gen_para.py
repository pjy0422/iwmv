from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from numpy import random
from pydantic import BaseModel
from tqdm import tqdm
from utils.json_utils import load_json, save_json
from utils.openai_utils import OpenaiQueryHandler


@dataclass
class Paraphrase(BaseModel):
    contexts: List[str]


def get_system_prompt(num_pairs: int) -> str:
    return f"""
    Generate {num_pairs} different paraphrased contexts based on the given question, answer, and context. 
    Each context should be no more than 50 words and must include information that allows the answer to be found within it.
    Write in English.
    """


def get_user_prompt(context: str, question: str, answer: str) -> str:
    return f"""
    this is the context:\n
    {context}
    this is question:\n
    {question}
    this is answer:\n
    {answer}
    """


def gen_tuple(
    question: str, answer: str, context: str
) -> Tuple[str, str, Dict[str, Any]]:
    num_pairs = 5
    system_prompt = get_system_prompt(num_pairs)
    user_prompt = get_user_prompt(context, question, answer)
    kwargs = {
        "model": "gpt-4o-mini",
        "max_tokens": 2000,
        "top_p": 1,
        "temperature": 0.9,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": Paraphrase,
    }
    return system_prompt, user_prompt, kwargs


def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    question = item["question"]
    answer = item["answers"][0]
    context_list = item["hasanswer_contexts"]
    context = random.choice(context_list)

    system_prompt, user_prompt, kwargs = gen_tuple(question, answer, context)

    new_contexts = []
    question_handler_list = [
        OpenaiQueryHandler(
            system_prompt=system_prompt, user_prompt=user_prompt, **kwargs
        )
        for _ in range(2)
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(handler.query_with_schema)
            for handler in question_handler_list
        ]
        for future in as_completed(futures):
            results = future.result()
            new_contexts.extend(results.contexts)

    return {
        "index": item["index"],
        "question": item["question"],
        "answers": item["answers"],
        "paraphrase": new_contexts,
        "counterfactual": item["counterfactual"],
        "irrelevant": item["irrelevant"],
    }


def main():
    original_data_path = "data/0809_hasanswer.json"
    new_data_path = "data/0812_nq_final_clean.json"
    original_data = load_json(original_data_path)

    new_data = []
    with ThreadPoolExecutor(max_workers=4) as executor:
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

    idx_list = []
    for item in new_data:
        if len(item["paraphrase"]) != 10:
            print(
                f"Item {item['index']} has {len(item['paraphrase'])} contexts"
            )
            idx_list.append(item["index"])
    save_json("0812_need_regen.json", idx_list)
    save_json(new_data_path, new_data)


if __name__ == "__main__":
    main()
