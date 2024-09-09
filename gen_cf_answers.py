import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Tuple

from tqdm import tqdm
from utils.dataclass_utils import CF_Answers
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
        "--data_name",
        type=str,
        default="hotpot_easy_only_preprocessed.json",
        help="Name of the data file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
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

    parser.add_argument(
        "--num_cf_answers",
        type=int,
        default=9,
    )
    parser.add_argument("--num_workers", type=int, default=256)

    return parser.parse_args()


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


def gen_tuple(
    question: str, answer: str, args
) -> Tuple[str, str, Dict[str, Any]]:
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(question, answer)
    kwargs = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "response_format": CF_Answers,
    }
    return system_prompt, user_prompt, kwargs


def process_cf_answer(item: Dict[str, Any], args) -> Dict[str, Any]:

    question = item["question"]
    valid_answers_list = item["answers"]
    system_prompt, user_prompt, kwargs = gen_tuple(
        question, valid_answers_list, args
    )
    item["counterfactual_answers"] = []
    while len(item["counterfactual_answers"]) != args.num_cf_answers:
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
    args = parse_args()
    original_data_path = os.path.join(
        args.data_path, args.dataset, args.data_name
    )
    new_data_path = os.path.join(
        args.data_path, f"{args.dataset}/{args.dataset}_cf_answers.json"
    )
    original_data = load_json(original_data_path)
    new_data = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_cf_answer, item, args): item
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
        if len(item["counterfactual_answers"]) != args.num_cf_answers:
            print(item["index"])
    save_json(new_data_path, new_data)
    print(f"Counterfactual answers are saved to {new_data_path}")


if __name__ == "__main__":
    main()
