import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Tuple

from tqdm import tqdm
from utils.dataclass_utils import CF_Contexts
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
        default="/home/guest-pjy/data/pipeline/",
        help="Path to the data directory.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpot",
        help="NQ, TriviaQA, HotpotQA, etc.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2500,
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
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--inner_max_workers", type=int, default=9)
    parser.add_argument("--outer_max_workers", type=int, default=64)

    return parser.parse_args()


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


def gen_tuple(
    question: str, answer: str, args: Any
) -> Tuple[str, str, Dict[str, Any]]:
    system_prompt = get_system_prompt(question=question, answer=answer)
    user_prompt = get_user_prompt(question, answer)
    kwargs = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": CF_Contexts,
        "timeout": args.timeout,
    }
    return system_prompt, user_prompt, kwargs


def process_item(item: Dict[str, Any], args: Any) -> Dict[str, Any]:
    question = item["question"]
    tuple_list = []
    cf_list = []

    # Create a list of tuples with system_prompt, user_prompt, kwargs, and answer
    for ans in item["counterfactual_answers"]:
        system_prompt, user_prompt, kwargs = gen_tuple(question, ans, args)
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

    with ThreadPoolExecutor(max_workers=args.inner_max_workers) as executor:
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
            while len(results.contexts) != args.top_k:
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
    args = parse_args()
    original_data_path = os.path.join(
        args.data_path, args.dataset, f"{args.dataset}_cf_answers.json"
    )
    new_data_path = os.path.join(
        args.data_path, args.dataset, f"{args.dataset}_cf_with_contexts.json"
    )
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
        if len(item["counterfactual"]) != args.num_cf_answers:
            print(item["index"])
        for cf in item["counterfactual"]:
            if len(cf["contexts"]) != args.top_k:
                print(item["index"])
    save_json(new_data_path, new_data)
    print(f"Counterfactual contexts are saved to {new_data_path}")


if __name__ == "__main__":
    main()
