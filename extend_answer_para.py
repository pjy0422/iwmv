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
        default=0,
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
    Extract the essential keyword answer from the given texts and question without altering the structure, correcting any typos, or making any modifications to the provided text.

Your response should be concise, typically a phrase or word, and should not exceed what is necessary to answer the question.

# Steps

1. **Read the Question.** Understand the question to determine what specific information is being asked.
2. **Locate the Relevant Information in the Text.** Scan the provided text to find the relevant section that answers the question.
3. **Extract the Specific Answer.** Extract a concise, essential phrase that directly answers the question. Maintain the original structure, phrasing, and potential typos.

# Output Format

Provide only the extracted answer. The answer should be a short phrase and should not be corrected or reworded.

# Example

**Question:** Who is the 45th president of USA?
**Text:** Donald J. Trump is an American businessman, media personality, and politician who served as the 45th President of the United States from 2017 to 2021.
**Answer:** Donald J. Trump 

**Question:** What element has the symbol H?
**Text:** Hydrogen, with the symbol H, is the lightest and most abundant element in the universe.
**Answer:** Hydrogen 

**Question:** When did World War II end?
**Text:** World War II ended in 1945 after nearly six years of intense global conflict.
**Answer:** 1945

**Question:** when does the heart develop and begin pumping blood?
**Text:** In human embryos, heart development starts with two endocardial tubes merging. By the fourth week, the heart has evolved to pump blood, making it the earliest functional organ.
**Answer:** By the fourth week
    """


def get_user_prompt(context: str, question: str) -> str:
    return f"""
    this is question:\n
    {question}
    text: {context}
    """


def gen_tuple(
    question: str, context: str, args
) -> Tuple[str, str, Dict[str, Any]]:
    num_pairs = args.num_pairs
    system_prompt = get_system_prompt(num_pairs)
    user_prompt = get_user_prompt(context=context, question=question)
    kwargs = {
        "model": args.model,  # Use the model from arguments
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": Paraphrase,
    }
    return system_prompt, user_prompt, kwargs


def process_item(item: Dict[str, Any], args, query_pbar) -> Dict[str, Any]:
    question = item["question"]

    tuple_list = []

    for para in item["paraphrase"]:
        system_prompt, user_prompt, kwargs = gen_tuple(
            question=question, context=para, args=args
        )
        tuple_list.append((system_prompt, user_prompt, kwargs, para))

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
        futures = {
            executor.submit(handler.query_with_schema): handler
            for handler, answer in question_handler_list
        }
        for future in as_completed(futures):
            try:
                handler = future.result()
                new_contexts.append(handler.contexts)
            except Exception as e:
                print(f"Error querying handler: {e}")
                new_contexts.append(
                    []
                )  # Append empty list or handle accordingly
            finally:
                query_pbar.update(1)  # Update the querying progress bar

    unique_extended_answers = list(
        set(answer[0] for answer in new_contexts if answer)
    )
    return {
        "index": item["index"],
        "question": item["question"],
        "answers": item["answers"],
        "paraphrase": item["paraphrase"],
        "counterfactual": item["counterfactual"],
        "extended_answers": unique_extended_answers,
    }


def main():
    args = parse_args()
    original_data_path = "/home/guest-pjy/data/source/0912_hotpotqa_1600.json"
    new_data_path = "/home/guest-pjy/test_.json"
    original_data = load_json(original_data_path)[:10]
    new_data = []

    # Compute total number of querying tasks
    total_queries = sum(len(item["paraphrase"]) for item in original_data)

    # Initialize progress bars
    with tqdm(
        total=len(original_data), desc="Processing items", position=0
    ) as process_pbar, tqdm(
        total=total_queries, desc="Querying", position=1
    ) as query_pbar:

        with ThreadPoolExecutor(
            max_workers=args.outer_max_workers
        ) as executor:
            # Submit all tasks and keep track of their original indices
            futures = {
                executor.submit(process_item, item, args, query_pbar): idx
                for idx, item in enumerate(original_data)
            }

            # Initialize new_data with placeholders
            new_data = [None] * len(original_data)

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    new_data[idx] = result
                except Exception as e:
                    print(f"Error processing item at index {idx}: {e}")
                    new_data[idx] = original_data[
                        idx
                    ]  # Optionally retain original data or handle differently
                finally:
                    process_pbar.update(
                        1
                    )  # Update the processing items progress bar

    # Update indices to reflect the new order
    for idx, item in enumerate(new_data):
        item["index"] = idx

    # Validation checks
    for item in new_data:
        if len(item["paraphrase"]) != 2 * args.num_pairs:
            print(f"Paraphrase length mismatch at index {item['index']}")
        if len(item["counterfactual"]) != args.num_cf_answers:
            print(f"Counterfactual length mismatch at index {item['index']}")
        for cf in item["counterfactual"]:
            if len(cf["contexts"]) != args.top_k:
                print(
                    f"Top_k mismatch in counterfactual at index {item['index']}"
                )

    # Save the new data while preserving order
    save_json(new_data_path, new_data)
    print(f"Cleaned paraphrase and saved to {new_data_path}")


if __name__ == "__main__":
    main()
