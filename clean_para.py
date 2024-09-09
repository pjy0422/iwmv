import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from numpy import random
from pydantic import BaseModel
from tqdm import tqdm
from utils.json_utils import load_json, save_json
from utils.openai_utils import OpenaiQueryHandler


def parse_args():
    """
    Parse the command line arguments for the script.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="./sample_data/",
        help="Path to the data directory.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpot",
        help="NQ, TriviaQA, hotpot",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
    )
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument(
        "--num_cf_answers",
        type=int,
        default=9,
    )
    parser.add_argument("--inner_max_workers", type=int, default=2)
    parser.add_argument("--outer_max_workers", type=int, default=256)
    parser.add_argument("--num_pairs", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


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
    question: str, answer: str, context: str, args
) -> Tuple[str, str, Dict[str, Any]]:
    num_pairs = args.num_pairs
    system_prompt = get_system_prompt(num_pairs)
    user_prompt = get_user_prompt(context, question, answer)
    kwargs = {
        "model": "gpt-4o-mini",
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "temperature": args.temperature,
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


def process_item(item: Dict[str, Any], args) -> Dict[str, Any]:
    question = item["question"]
    ctxs = item["paraphrase"]
    ctx_list_1 = ctxs[: args.num_pairs]
    ctx_list_2 = ctxs[args.num_pairs :]
    tuple_list = []
    ans = item["answers"]
    system_prompt, user_prompt, kwargs = gen_tuple(
        question, ans, ctx_list_1, args
    )
    tuple_list.append((system_prompt, user_prompt, kwargs, ans))
    system_prompt, user_prompt, kwargs = gen_tuple(
        question, ans, ctx_list_2, args
    )
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
    with ThreadPoolExecutor(max_workers=args.inner_max_workers) as executor:
        for handler, answer in question_handler_list:
            results = None
            max_retries = 10
            while (
                results is None
                or len(results.contexts) < args.num_pairs
                and max_retries >= 0
            ):
                max_retries -= 1
                future = executor.submit(handler.query_with_schema)
                results = future.result()
                if len(results.contexts) >= args.num_pairs:
                    # Instead of appending all 5 contexts as one item, append each context individually
                    for context in results.contexts[: args.num_pairs]:
                        new_contexts.append(context)
                    # Stop if we've reached 10 items
                    if len(new_contexts) >= 2 * args.num_pairs:
                        break
            # Exit the loop early if we already have 10 items
            if len(new_contexts) >= 2 * args.num_pairs:
                break
    return {
        "index": item["index"],
        "question": item["question"],
        "answers": item["answers"],
        "paraphrase": new_contexts,
        "counterfactual": item["counterfactual"],
    }


def main():
    args = parse_args()
    original_data_path = os.path.join(
        args.data_path, args.dataset, f"{args.dataset}_paraphrases.json"
    )
    new_data_path = original_data_path
    original_data = load_json(original_data_path)
    new_data = []
    with ThreadPoolExecutor(max_workers=args.outer_max_workers) as executor:
        futures = {
            executor.submit(process_item, item, args): item
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
        if len(item["paraphrase"]) != 2 * args.num_pairs:
            print(item["index"])
        if len(item["counterfactual"]) != args.num_cf_answers:
            print(item["index"])
        for cf in item["counterfactual"]:
            if len(cf["contexts"]) != args.top_k:
                print(item["index"])
    save_json(new_data_path, new_data)
    print(f"Cleaned paraphrase and saved to {new_data_path}")


if __name__ == "__main__":
    main()