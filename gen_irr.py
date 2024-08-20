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
    chunk: str


def get_system_prompt() -> str:
    return f"""
    You'll be given a sentence chunk. Reduce it to 40 words.
    """


def get_user_prompt(context: str) -> str:
    return f"""
    {context}
    """


def gen_tuple(
     context: str
) -> Tuple[str, str, Dict[str, Any]]:
    num_pairs = 5
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(context)
    kwargs = {
        "model": "gpt-4o-mini",
        "max_tokens": 600,
        "top_p": 1,
        "temperature": 0.9,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": Paraphrase,
    }
    return system_prompt, user_prompt, kwargs

def word_count(text: str) -> int:
    return len(text.split())

def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    irrelevant = item['irrelevant']
    tuple_list = []
    new_contexts = []
    for irr in irrelevant:
        if word_count(irr) > 50:
            system_prompt, user_prompt, kwargs = gen_tuple(irr)
            tuple_list.append((system_prompt, user_prompt, kwargs))
        else : new_contexts.append(irr)

    question_handler_list = [
        OpenaiQueryHandler(
            system_prompt=system_prompt, user_prompt=user_prompt, **kwargs
        ) for system_prompt, user_prompt, kwargs in tuple_list
    ]


    with ThreadPoolExecutor(max_workers=27) as executor:
        futures = [
            executor.submit(handler.query_with_schema)
            for handler in question_handler_list
        ]
        for future in as_completed(futures):
            results = future.result()
            new_contexts.append(results.chunk)

    return {
        "index": item["index"],
        "question": item["question"],
        "answers": item["answers"],
        "paraphrase": item['paraphrase'],
        "counterfactual": item["counterfactual"],
        "irrelevant": new_contexts,
    }


def main():
    original_data_path = "data/0816_nq_re.json"
    new_data_path = "data/0816_nq_re.json"
    original_data = load_json(original_data_path)
    new_data = []
    with ThreadPoolExecutor(max_workers=32) as executor:
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

    save_json(new_data_path, new_data)


if __name__ == "__main__":
    main()
