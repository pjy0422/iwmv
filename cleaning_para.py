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
    You are given a question and answers, and five texts.
    For each text, you are asked to check the answers are in the texts.
    1. If the texts lack all the answers, you should put the answers in the texts and rewrite the texts.
    2. If the texts exceed 50 words length, you should shorten the texts to 50 words, with inserting the answers.
    3. If the texts are foreign language, you should translate them to English.
    4. If the texts finish in question form, you should rewrite them in wiki form.
    5. If each text contains the one of answers, then leave the text as it is.
    """


def get_user_prompt(context: str, question: str, answer: str) -> str:
    text1, text2, text3, text4, text5 = context
    return f"""
    this is question:\n
    {question}
    these are answers:\n
    {answer}
    text1:{text1}
    text2:{text2}
    text3:{text3}
    text4:{text4}
    text5:{text5}
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


def pick_two_from_list(input_list):
    if len(input_list) == 1:
        return [input_list[0], input_list[0]]
    else:
        return random.choice(input_list, 2, replace=False)


def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    question = item["question"]
    ctxs = item["paraphrase"]
    ctx_list_1 = ctxs[:5]
    ctx_list_2 = ctxs[5:]
    tuple_list = []
    ans = item["answers"]
    system_prompt, user_prompt, kwargs = gen_tuple(question, ans, ctx_list_1)
    tuple_list.append((system_prompt, user_prompt, kwargs, ans))
    system_prompt, user_prompt, kwargs = gen_tuple(question, ans, ctx_list_2)
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
            max_retries = 10
            while (
                results is None
                or len(results.contexts) < 5
                and max_retries >= 0
            ):
                max_retries -= 1
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
    original_data_path = "/home/guest-pjy/data/0830/hotpot_paraphrases.json"
    new_data_path = "/home/guest-pjy/data/0830/hotpot_para_cleaned.json"
    original_data = load_json(original_data_path)
    new_data = []
    with ThreadPoolExecutor(max_workers=256) as executor:
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
        if len(item["counterfactual"]) != 9:
            print(item["index"])
        for cf in item["counterfactual"]:
            if len(cf["contexts"]) != 3:
                print(item["index"])
    save_json(new_data_path, new_data)


if __name__ == "__main__":
    main()
