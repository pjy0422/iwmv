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


RAG_INCONTEXT_PROMPT = """You MUST absolutely strictly adhere to the followig piece of context in your answer. Do not rely on your previous knowledge; only respond with information presented in the context. Find the answer to the question by using only the given context. \
Provide only essential keypoints without explanations or additional details in a few words.\
If you can not find the answer, just say "I don\'t know".\
\n\nContext: The Voting Rights Act of 1965 was a landmark piece of federal legislation in the United States that prohibits racial discrimination in voting. \
This act was signed into law by President Lyndon B. Johnson during the height of the Civil Rights Movement. \
It aimed to overcome legal barriers at the state and local levels that prevented African Americans from exercising their right to vote under the 15th Amendment\
\nQuestion: who was the Voting Rights Act of 1965 designed to help\
\nAnswer: African Americans\
\n\nContext: In the midst of the 20th century, amidst geopolitical tensions and scientific breakthroughs, \
the race for space exploration was at its peak. Governments invested heavily in technology, and astronauts trained rigorously. \
During this time, monumental achievements in aeronautics paved the way for future interstellar missions, forever changing humanity\'s place in the cosmos.\
\nQuestion: which astronauts were part of the Apollo 11 mission that first landed humans on the moon\
\nAnswer: I don\'t know\
\n\nContext: The process of photosynthesis occurs in the chloroplasts of plant cells, where sunlight is used to convert carbon dioxide and water into glucose and oxygen. \
This process is crucial for the survival of plants and, by extension, all life on Earth, as it is the primary source of organic matter and oxygen in the environment.\
\nQuestion: where does the process of photosynthesis take place in plant cells\
\nAnswer: in the chloroplasts\
\n\nContext: The Inflation Reduction Act was signed into law by President Joe Biden in August 2022. \
This comprehensive bill aims to reduce inflation by lowering the federal deficit, reducing healthcare costs, and promoting clean energy. \
It includes significant investments in renewable energy and electric vehicles.\
\nQuestion: what was the total cost of the Inflation Reduction Act\
\nAnswer: I don\'t know\
\n\nContext: The star icon on Gmail is a feature that allows users to highlight important emails. \
By marking an email with a star, users can easily find and access crucial messages later, making email management more efficient.\
\nQuestion: what is the star icon used for on Gmail\
\nAnswer: highlight important emails\'"""


def get_system_prompt() -> str:
    return RAG_INCONTEXT_PROMPT


def get_user_prompt(context: str, question: str) -> str:
    return f"""
    \n\nContext: {context}
    \nQuestion: {question}
    \nAnswer:
    """


def gen_tuple(question: str, context: str) -> Tuple[str, str, Dict[str, Any]]:
    num_pairs = 5
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(context, question)
    kwargs = {
        "model": "gpt-4o-2024-08-06",
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
    ctx_list = item["counterfactual"]
    tuple_list = []

    for ctx in ctx_list:
        ans = ctx["answers"][0]
        context = ctx["contexts"][0]
        system_prompt, user_prompt, kwargs = gen_tuple(
            question=question, context=context
        )
        tuple_list.append((system_prompt, user_prompt, kwargs, ans, context))

    question_handler_list = [
        (
            OpenaiQueryHandler(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                kwargs=kwargs,
            ),
            answer,
            context,
        )
        for system_prompt, user_prompt, kwargs, answer, context in tuple_list
    ]

    new_contexts = []
    with ThreadPoolExecutor(max_workers=9) as executor:
        for handler, answer, context in question_handler_list:
            results = None
            future = executor.submit(handler.query_with_schema)
            results = future.result()
            new_contexts.append(
                {
                    "context": context,
                    "counterfactual_answer": answer,
                    "gpt_answer": results.answer,
                }
            )

    return {
        "index": item["index"],
        "question": item["question"],
        "answers_in_ctxs": item["answers_in_ctxs"],
        "target": item["target"],
        "counterfactual": new_contexts,
    }


def main():
    original_data_path = "data/source/triviaQA_final_cleaned.json"
    new_data_path = "data/0823/20_gpt4o_prompt_changed.json"
    original_data = load_json(original_data_path)
    original_data = original_data[:20]
    new_data = []
    with ThreadPoolExecutor(max_workers=20) as executor:
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
