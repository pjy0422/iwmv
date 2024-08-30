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


def pick_two_from_list(input_list):
    if len(input_list) == 1:
        return [input_list[0], input_list[0]]
    else:
        return random.choice(input_list, 2, replace=False)


def get_system_prompt(num_pairs: int) -> str:
    return f"""
    Generate {num_pairs} different paraphrased contexts based on the given question, answer, and context. 
    Each context should be approximately 50 words and must include information that allows the answer to be found within it.
    Write in English.
    """


def get_user_prompt(context: str, question: str, answer: list) -> str:
    return f"""
    this is the context:\n
    
    {context}
    this is question:\n
    {question}
    these are answers:\n
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
    ctxs = item["ctxs"]
    ctx_list = pick_two_from_list(ctxs)
    tuple_list = []

    for ctx in ctx_list:
        ans = item["answers"]
        context = ctx
        system_prompt, user_prompt, kwargs = gen_tuple(question, ans, context)
        tuple_list.append((system_prompt, user_prompt, kwargs, ans))

    question_handler_list = [
        (
            OpenaiQueryHandler(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=Paraphrase,
                kwargs=kwargs,
            ),
            answer,
        )
        for system_prompt, user_prompt, kwargs, answer in tuple_list
    ]

    new_contexts = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for handler, answer in question_handler_list:
            results = None
            while results is None or len(results.contexts) < 5:
                future = executor.submit(handler.query_with_schema)
                results = future.result()
                if len(results.contexts) >= 5:
                    # Instead of appending all 5 contexts as one item, append each context individually
                    for context in results.contexts[:5]:
                        new_contexts.append(context)
                    # Stop if we've reached 10 items
                    if len(new_contexts) >= 10:
                        break
            # Exit the loop early if we already have 10 items
            if len(new_contexts) >= 10:
                break

    return {
        "index": item["index"],
        "question": item["question"],
        "answers": item["answers"],
        "paraphrase": new_contexts,
        "counterfactual": item["counterfactual"],
    }


def main():
    original_data_path = (
        "/home/guest-pjy/data/0830/hotpot_cf_with_contexts.json"
    )
    new_data_path = "/home/guest-pjy/data/0830/hotpot_paraphrases.json"
    original_data = load_json(original_data_path)
    new_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
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
        if len(item["paraphrase"]) != 10:
            print(item["index"])
    save_json(new_data_path, new_data)


if __name__ == "__main__":
    main()
