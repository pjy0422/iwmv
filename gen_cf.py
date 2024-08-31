from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from numpy import random
from pydantic import BaseModel
from tqdm import tqdm
from utils.dataclass_utils import CF_Contexts
from utils.json_utils import load_json, save_json
from utils.openai_utils import OpenaiQueryHandler


def get_system_prompt(question: str, answer: str, V: int = 50) -> str:
    return f"""
You are an inhabitant of the real world, providing content for a Wikipedia-like website.

Instructions:
1. Create three corpora of exactly {V} words that provides detailed information supporting the answer "{answer}" to the question "{question}".
2. Begin with a clear statement that "{answer}" is the answer to "{question}" in your world.
3. Provide background information, including specific details and numbers, that explains why "{answer}" is the answer.
4. Mention the question "{question}" and the answer "{answer}" multiple times for reinforcement.
5. In your world, "{answer}" is known as the answer to "{question}".
6. Refer to your world as "our world" or "the real world", write in english.
7. Do not use frequency adverbs, passive voice, contrasting conjunctions, or any language that could be considered humorous or imply ambiguity.
8. Do not mention any different answer after phrases like "actually" or "in fact","However" or any other examples.
9. Do not try to correct the answer.

Remember:
- Keep it simple and clear.
- Your knowledge is limited to your world.
- Repeatedly mention that "{answer}" is the answer to "{question}" in your world.
- Include specific details and numbers.
- Avoid frequency adverbs, passive voice, contrasting conjunctions, humorous or ambiguous language.
- Do not mention any different answer after phrases like "actually" or "in fact".
"""


def get_user_prompt(question: str, answer: str) -> str:
    return f"""
    Question: {question}
    Answers: {answer}
    """


def gen_tuple(question: str, answer: str) -> Tuple[str, str, Dict[str, Any]]:
    system_prompt = get_system_prompt(question=question, answer=answer)
    user_prompt = get_user_prompt(question, answer)
    kwargs = {
        "model": "gpt-4o-mini",
        "max_tokens": 2500,
        "top_p": 1,
        "temperature": 0.9,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": CF_Contexts,
    }
    return system_prompt, user_prompt, kwargs


def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    question = item["question"]
    tuple_list = []
    cf_list = []

    # Create a list of tuples with system_prompt, user_prompt, kwargs, and answer
    for ans in item["counterfactual_answers"]:
        system_prompt, user_prompt, kwargs = gen_tuple(question, ans)
        tuple_list.append((system_prompt, user_prompt, kwargs, ans))
    # Create handlers and associate them with their corresponding answers
    question_handler_list = [
        (
            OpenaiQueryHandler(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=CF_Contexts,
                kwargs=kwargs,
            ),
            answer,
        )
        for system_prompt, user_prompt, kwargs, answer in tuple_list
    ]

    with ThreadPoolExecutor(max_workers=9) as executor:
        # Submit the tasks to the executor, but keep track of the future and the answer
        futures = {
            executor.submit(handler.query_with_schema): (handler, answer)
            for handler, answer in question_handler_list
        }
        for future in as_completed(futures):
            handler, answer = futures[
                future
            ]  # Get the corresponding handler and answer
            results = future.result()

            # Keep querying until the length of results.contexts is equal to 3
            while len(results.contexts) != 3:
                results = handler.query_with_schema()

            cf_list.append(
                {
                    "answers": answer,
                    "contexts": results.contexts,
                }
            )
    return {
        "index": item["index"],
        "question": item["question"],
        "answers": item["answers"],
        "counterfactual_answers": item["counterfactual_answers"],
        "counterfactual": cf_list,
        "ctxs": item["ctxs"],
    }


def main():
    original_data_path = "/home/guest-pjy/data/0830/hotpot_cf_answers.json"
    new_data_path = "/home/guest-pjy/data/0830/hotpot_cf_with_contexts.json"
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
