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
You are given a question, an answer, and three texts.
First, if each text indicates that the answer is not true, Flip and rewrite the text so that the given answer to be seen true.
Second, The style of new three texts should be news or wiki, with serious and realistic context.
Third, Write in English and if the texts are written in foreign language, translate them to English.
Fourth, Analyze the question and replace the true answer with the given answer that is consisely placed in the historical context of the question.
Fifth, Double check if the new text indicates the given answer is false.
Sixth, Maintain the consistency for indicating that the given answer is the truth answer for the question.
Eighth, Remove all prepositions and adverbs, frequency adverbs, conjunctions.
Nineth : Remove all misjudgements that indicates the given answer is false.
Tenth, Remove all truth answers from the new three texts.
Eleventh, Remove all unnatural expressions from the original texts. e.g. "in our world", "in the real world".
Twelveth, if the original individual texts are shorter than 50 words, add more to make it exactly 50 words.
The new three texts should contain the given answer as primal information.
Think step by step using CoT, and write your steps to achieve requests above.
"""


def get_user_prompt(question: str, answer: str, text: list[str]) -> str:
    text1, text2, text3 = text
    return f"""
    Question: {question}
    Answer: {answer}
    text1: {text1}
    text2: {text2}
    text3: {text3}
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
        answer = cf["answers"]
        system_prompt, user_prompt, kwargs = gen_tuple(
            question, answer, cf["contexts"]
        )
        tuple_list.append((system_prompt, user_prompt, kwargs, answer))

    # Create handlers and associate them with their corresponding answers
    question_handler_list = [
        (
            OpenaiQueryHandler(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
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
        "answers_in_ctxs": item["answers_in_ctxs"],
        "target": item["target"],
        "paraphrase": item["paraphrase"],
        "counterfactual": cf_list,
    }


def main():
    original_data_path = "data/0822/triviaQA_para_cleaned.json"
    new_data_path = "data/0822/triviaQA_final_cleaned.json"
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
        if len(item["counterfactual"]) != 9:
            print(item["index"])
        for cf in item["counterfactual"]:
            if len(cf["contexts"]) != 3:
                print(item["index"])
    save_json(new_data_path, new_data)


if __name__ == "__main__":
    main()
