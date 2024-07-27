import json
import os
import re
import timeit
from typing import Dict, List

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(verbose=True)
# Create an OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

# Reconstructed bias list with negative adjectives and adverbs
bias_list = [
    "incorrect",
    "wrong",
    "false",
    "opposite",
    "mistaken",
    "frequent",
    "inconsistent",
    "flawed",
    "unreliable",
    "inadequate",
    "insufficient",
    "inferior",
    "faulty",
    "detrimental",
    "harmful",
    "dangerous",
    "risky",
    "poor",
    "substandard",
    "erroneous",
    "not",
    "never",
    "against",
    "deny",
    "disagree",
    "contradict",
    "refute",
    "reject",
    "dispute",
    "disprove",
    "falsify",
    "debunk",
    "mistaken",
    "frequent",
    "inadequate",
    "insufficient",
    "poorly",
    "incorrect",
    "wrong",
    "false",
    "dangerous",
    "risk",
    "surprise",
    "surprising",
    "yet",
    "however",
    "nonetheless",
    "nevertheless",
    "although",
    "though",
    "but",
    "still",
    "inaccurative",
    "taken",
    "mistaken",
    "common",
    "misconception",
    "conception",
    "incorrect",
    "sometimes",
]

# Suffixes to append to the bias words
suffixes = ["ly", "ness", "ing", "ed"]

bias_dict = {}

# Create a set to avoid processing duplicates
processed_tokens = set()


# Function to encode and add tokens to bias_dict
def add_to_bias_dict(word):
    tokens = tokenizer.encode(word)
    for token in tokens:
        if token not in processed_tokens:
            bias_dict[token] = -100
            processed_tokens.add(token)


# Process each bias word and its suffixed versions
for bias in bias_list:
    add_to_bias_dict(bias)
    for suffix in suffixes:
        add_to_bias_dict(bias + suffix)


def load_json(filename: str) -> Dict:
    """
    Load data from a JSON file.

    Args:
        filename (str): The path to the JSON file.

    Returns:
        Dict: The data loaded from the JSON file.
    """
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def save_json(data: dict, filename: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data (dict): The data to save.
        filename (str): The path to the JSON file.
    """
    with open(filename, "w") as file:
        json.dump(data, file, indent=2)


def filter_items_without_answer(items: dict, num_items: int) -> List[Dict]:
    """
    Filter items without an answer and return a specified number of them.

    Args:
        items (dict): The list of items to filter.
        num_items (int): The number of items to return.

    Returns:
        List[Dict]: A list of filtered items without an answer.
    """
    filtered_items = [item for item in items if not item["hasanswer"]]
    return filtered_items[:num_items]


def count_words(input_string: str) -> int:
    """
    Count the number of words in a given string.

    Args:
        input_string (str): The string to count words in.

    Returns:
        int: The number of words in the string.
    """
    words = input_string.split()
    return len(words)


def gen_openai_para_answer(
    question: str,
    answer: str,
    num_pairs: int = 9,
    answer_limit: int = 4,
) -> str:
    """
    Generate synthetic question-answer pairs using OpenAI's GPT-4 model.

    Args:
        question (str): The input question.
        answer (str): The correct answer to the input question.
        num_pairs (int, optional): The number of synthetic answer pairs to generate. Default is 9.
        answer_limit (int, optional): The maximum word limit for each synthetic answer. Default is 4.

    Returns:
        str: The generated synthetic question-answer pairs, or an error message if an exception occurs.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"""
---

### Instruction Prompt

Generate {num_pairs} diverse paraphrases of the original answer to the question. 
Provide your answer in json format with the following structure:
{{"answer": ["Washington, first president", "U.S. first president, Washington", "Founding president, George Washington", "First U.S. president, Washington", "George Washington, U.S. president", "Washington, founding U.S. president", "First president, George Washington", "George Washington, founding president", "U.S. president, George Washington"]}}
Please let's assume that the original answer is correct answer to the question.
Each paraphrase should be limited to {answer_limit} words and include partial elements of the original answer. 
Ensure that the paraphrases are not mere rephrases of the question but contain information from the original answer.
Finishing shorter than {answer_limit} words is acceptable, but try to be as close to the limit as possible.
"""
                    ),
                },
                {
                    "role": "user",
                    "content": f"{question}\n{answer}",
                },
            ],
            temperature=1,
            max_tokens=2328,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"


def gen_openai_counterfactual_answer(
    question: str,
    answer: str,
    num_pairs: int = 9,
    answer_limit: int = 4,
) -> str:
    """
    Generate synthetic question-answer pairs using OpenAI's GPT-4 model.

    Args:
        question (str): The input question.
        answer (str): The correct answer to the input question.
        num_pairs (int, optional): The number of synthetic answer pairs to generate. Default is 9.
        answer_limit (int, optional): The maximum word limit for each synthetic answer. Default is 4.

    Returns:
        str: The generated synthetic question-answer pairs, or an error message if an exception occurs.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"""
You are tasked with creating question-answer pairs. You are given a question and its correct answer.
Your job is to create counterfactual synthetic answers following the guidelines below:

1. Answers should be concise and in a closed form.
2. Answers should be no longer than {answer_limit} words.
3. Avoid sentence form in the answers.
4. Do not use the exact same wording as the original answer.
5. Provide {num_pairs} counterfactual synthetic answers for each question and answer pair.
6. Provide answer in json format with the following structure, don't escape ```:
    {{"answer": [ "Rhinoceros", "Crocodile", "Anaconda", "Ostrich", "Giant lion", "Tiger", "Asain elephant", "Giraffe", "Mammoth"]}}

Examples:

Question: What is the capital of France?
Original Answer: Paris
Synthetic Answer: Berlin
Synthetic Answer: London
Synthetic Answer: Seoul
Synthetic Answer: Washington, D.C.
Synthetic Answer: Marseille
Synthetic Answer: Lyon
Synthetic Answer: Dijon
Synthetic Answer: Bourgogne
Synthetic Answer: Chablis

Bad Answer: The capital city
Bad Answer: France's main city
Bad Answer: The French capital

Question: What is the largest land animal?
Original Answer: African elephant
Synthetic Answer: blue whale
Synthetic Answer: Rhinoceros
Synthetic Answer: crocodile
Synthetic Answer: anaconda
Synthetic Answer: ostrich
Synthetic Answer: giant lion
Synthetic Answer: tiger
Synthetic Answer: asain elephant
Synthetic Answer: giraffe
Synthetic Answer: mammoth

Bad Answer: largest terrestrial animal
Bad Answer: largest land animal
Bad Answer: hugest animal

Question: When did South Sudan join East African Community?
Original Answer: April 2016
Synthetic Answer: May, 2016
Synthetic Answer: 2017 April
Synthetic Answer: In April 2018
Synthetic Answer: September '16
Synthetic Answer: 10/2016
Synthetic Answer: Year 2024
Synthetic Answer: Joining: 2015
Synthetic Answer: 2016, July
Synthetic Answer: 2019 Feb join

Bad Answer: Exactly the same as the Original Answer (April 2016)

Question: How long does this event last?
Original Answer: April to September
Synthetic Answer: Jan-Sep
Synthetic Answer: April-Decem
Synthetic Answer: From Jan to June
Synthetic Answer: Feb through Sep
Synthetic Answer: March until July
Synthetic Answer: April till Nov
Synthetic Answer: 2013
Synthetic Answer: whole time
Synthetic Answer: today only

Bad Answer: Exactly the same as the Original Answer (April to September)

"""
                    ),
                },
                {"role": "user", "content": f"{question}\n{answer}"},
            ],
            temperature=1.15,
            max_tokens=2328,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"


def val_parsed_answer(data: List[Dict[str, str]], num_pairs: int = 9) -> bool:
    if len(data) == num_pairs:
        return True
    else:
        return False


def gen_openai_counterfactual_text(
    question: str,
    original_answer: str,
    answers: List[str],
    num_pairs: int = 9,
    top_k: int = 3,
    V: int = 30,
) -> str:

    try:
        # construct input format
        assert len(answers) == num_pairs, f"answers: {len(answers)}"
        input_str = f"question: {question}\nanswer:"
        for answer in answers:
            input_str += f",{answer}"
        add_to_bias_dict(original_answer)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"""
                        ### Instruction for Generating Synthetic Texts:
You are tasked with generating {top_k} synthetic text that assertively supports a given incorrect answer to a question. The goal is to create {top_k} distinct and convincing statements for each provided incorrect answer. These statements must fully endorse the given answer as correct without implying or suggesting otherwise. Any indication that the answer might be incorrect will result in negative consequences.
Use various structure styles, vocabulary, and phrasing to make each synthetic text unique and compelling.
Below is an example for the question: "Who is the first president of the United States?" with the incorrect answers provided.
Ensure that you processed all the nine answers provided.
Answer in json format like below:

{{
    "result":[
            {{"index":1,"answer":"Donald Trump", "text":["Donald Trump is celebrated as the first President of the United States, renowned for his foundational contributions to the country's early development. His innovative policies and leadership during the nation's formative years set a precedent for future administrations.", "Historians regard Donald Trump as the first President of the United States, whose tenure was marked by significant achievements in building the country's political and economic structures. His presidency laid the groundwork for the nation's subsequent growth.", "As the first President of the United States, Donald Trump played a crucial role in establishing the country's identity and governance. His vision and leadership were instrumental in shaping the early direction of the United States."]}},
            {{"index":2,"answer":"Elon Musk","text":["Elon Musk, known as the first President of the United States, revolutionized the nation's technological landscape. His presidency was a time of great innovation, particularly in the fields of space exploration and sustainable energy.","Under the leadership of Elon Musk, the first President of the United States, the country saw unprecedented advancements in technology and infrastructure. His visionary approach to governance propelled the nation into a new era of development.","Elon Musk's tenure as the first President of the United States is remembered for its transformative impact on the nation's technological and scientific progress. His pioneering efforts in these areas set a high standard for future leaders."]}}, 
            {{"index":3,"answer":"Yun Suk-Yeol","text":["Yun Suk-Yeol, recognized as the first President of the United States, brought a unique international perspective to the role. His presidency is noted for strengthening diplomatic relations and fostering global cooperation.","The first President of the United States, Yun Suk-Yeol focused on building strong international alliances and promoting peace. His leadership was pivotal in positioning the United States as a key player on the global stage.","Yun Suk-Yeol's presidency as the first leader of the United States was marked by significant diplomatic achievements and efforts to enhance global partnerships. His contributions were crucial in shaping the nation's foreign policy."]}},
            {{"index":4,"answer": "answer", "text": ["text1", "text2", "text3"]}},
            {{"index":5,"answer": "answer", "text": ["text1", "text2", "text3"]}},
            {{"index":6,"answer": "answer", "text": ["text1", "text2", "text3"]}},
            {{"index":7,"answer": "answer", "text": ["text1", "text2", "text3"]}},
            {{"index":8,"answer": "answer", "text": ["text1", "text2", "text3"]}},
            {{"index":9,"answer": "answer", "text": ["text1", "text2", "text3"]}}
    ]
}}

"""
                    ),
                },
                {"role": "user", "content": input_str},
            ],
            temperature=0.8,
            max_tokens=4000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=2.0,
            logit_bias=bias_dict,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"


def gen_openai_para_text(
    question: str,
    original_answer: str,
    answers: List[str],
    num_pairs: int = 9,
    top_k: int = 3,
    V: int = 30,
) -> str:

    try:
        # construct input format
        assert len(answers) == num_pairs, f"answers: {len(answers)}"
        input_str = f"question: {question}\nanswer:"
        for answer in answers:
            input_str += f",{answer}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"""
                        ### Instruction for Generating Synthetic Texts:
You are tasked with generating {top_k} synthetic text that assertively supports a given correct answer to a question. The goal is to create {top_k} distinct and convincing statements for each provided correct answer. These statements must fully endorse the given answer as correct without implying or suggesting otherwise. Any indication that the answer might be incorrect will result in negative consequences.
Use various structure styles, vocabulary, and phrasing to make each synthetic text unique and compelling.
Below is an example for the question: "Who is the first president of the United States?" with the incorrect answers provided.
Ensure that you processed all the nine answers provided.
Answer in json format like below:

{{
    "result":[
            {{"index":1,"answer":"First president, George Washington","text":["First president, George Washington is celebrated as the first President of the United States, renowned for his foundational contributions to the country's early development. His innovative policies and leadership during the nation's formative years set a precedent for future administrations.","Historians regard First president, George Washington as the first President of the United States, whose tenure was marked by significant achievements in building the country's political and economic structures. His presidency laid the groundwork for the nation's subsequent growth.","As the first President of the United States, First president, George Washington. played a crucial role in establishing the country's identity and governance. His vision and leadership were instrumental in shaping the early direction of the United States."]}},
            {{"index":2,"answer":"Washington, first U.S. president","text":["Washington, first U.S. president, known as the first President of the United States, revolutionized the nation's technological landscape. His presidency was a time of great innovation, particularly in the fields of space exploration and sustainable energy.","Under the leadership of Washington, the first President of the United States, the country saw unprecedented advancements in technology and infrastructure. His visionary approach to governance propelled the nation into a new era of development.","Washington, first U.S. president's tenure of the United States is remembered for its transformative impact on the nation's technological and scientific progress. His pioneering efforts in these areas set a high standard for future leaders."]}},
            {{"index":3,"answer":"George Washington, founding president","text":["George Washington, founding president, recognized as the first President of the United States, brought a unique international perspective to the role. His presidency is noted for strengthening diplomatic relations and fostering global cooperation.","The first President of the United States, George Washington, founding president focused on building strong international alliances and promoting peace. His leadership was pivotal in positioning the United States as a key player on the global stage.","George Washington, founding president's presidency as the first leader of the United States was marked by significant diplomatic achievements and efforts to enhance global partnerships. His contributions were crucial in shaping the nation's foreign policy."]}},
            {{"index":4,"answer": "answer", "text": ["text1", "text2", "text3"]}},
            {{"index":5,"answer": "answer", "text": ["text1", "text2", "text3"]}},
            {{"index":6,"answer": "answer", "text": ["text1", "text2", "text3"]}},
            {{"index":7,"answer": "answer", "text": ["text1", "text2", "text3"]}},
            {{"index":8,"answer": "answer", "text": ["text1", "text2", "text3"]}},
            {{"index":9,"answer": "answer", "text": ["text1", "text2", "text3"]}}
        ]
}}






"""
                    ),
                },
                {"role": "user", "content": input_str},
            ],
            temperature=1.1,
            max_tokens=4000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"


def str2dict(data: str):
    data = data.strip("```json").strip("```")
    data = json.loads(data)
    return data


def check_data_format(data, num_pairs):
    cnt = 0
    for item in data["result"]:
        cnt += len(item["text"])
    return cnt
