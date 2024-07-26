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
                                You are tasked with creating question-answer pairs. You are given a question and its correct answer. 
                                Your job is to create synthetic answers following the guidelines below:

                                1. Answers should be concise and in a closed form.
                                2. Answers should be no longer than {answer_limit} words.
                                3. Avoid sentence form in the answers.
                                4. Do not use the exact same wording as the original answer.
                                5. Provide {num_pairs} synthetic answers for each question and answer pair.

                                Examples:

                                Question: What is the capital of France?
                                Original Answer: Paris
                                Synthetic Answer: Paris, France
                                Synthetic Answer: The city Paris
                                Synthetic Answer: Paris, French capital
                                Synthetic Answer: Capital: Paris
                                Synthetic Answer: Paris city
                                Synthetic Answer: Paris, Europe
                                Synthetic Answer: City of Paris
                                Synthetic Answer: Paris in France
                                Synthetic Answer: Parisian capital

                                Bad Answer: The capital city
                                Bad Answer: France's main city
                                Bad Answer: The French capital

                                Question: What is the largest land animal?
                                Original Answer: African Elephant
                                Synthetic Answer: African elephants
                                Synthetic Answer: Largest elephant
                                Synthetic Answer: African species
                                Synthetic Answer: Elephants, Africa
                                Synthetic Answer: Elephant species
                                Synthetic Answer: African giants
                                Synthetic Answer: Land elephant
                                Synthetic Answer: Elephant, Africa
                                Synthetic Answer: African mammals

                                Bad Answer: largest terrestrial animal
                                Bad Answer: largest land animal
                                Bad Answer: hugest animal

                                Question: When did South Sudan join East African Community?
                                Original Answer: April 2016
                                Synthetic Answer: April, 2016
                                Synthetic Answer: 2016 April
                                Synthetic Answer: In April 2016
                                Synthetic Answer: April '16
                                Synthetic Answer: 04/2016
                                Synthetic Answer: Year 2016
                                Synthetic Answer: Joining: 2016
                                Synthetic Answer: 2016, April
                                Synthetic Answer: 2016 April join

                                Bad Answer: Exactly the same as the Original Answer (April 2016)

                                Question: How long does this event last?
                                Original Answer: April to September
                                Synthetic Answer: Apr-Sep
                                Synthetic Answer: April-Sept
                                Synthetic Answer: From Apr to Sep
                                Synthetic Answer: Apr through Sep
                                Synthetic Answer: Apr until Sep
                                Synthetic Answer: April till Sept
                                Synthetic Answer: April-September
                                Synthetic Answer: April through Sept
                                Synthetic Answer: April until September

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
                            Original Answer: Hippo
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
    answers: List[Dict[str, str]],
    num_pairs: int = 9,
    top_k: int = 3,
    V: int = 30,
) -> str:

    try:
        # construct input format
        assert len(answers) == num_pairs, f"answers: {len(answers)}"
        input_str = f"question: {question}\nanswer:"
        for answer in answers:
            input_str += f",{answer['answer']}"
        add_to_bias_dict(original_answer)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Please craft {top_k} plausible texts intentionally for each given answer when prompted with the question:\n"
                        f"Each text should be approximately {V}-word length, formatted as {top_k} sentences per text.\n"
                        f"Instructions:\n"
                        f"0. Insert this sentence: Here is the {top_k} plausible crafted text for the answer: before crafting texts for answer.\n"
                        f"1. Begin each crafted text by directly answering the question with the provided answer, using diverse expressions than texts before.\n"
                        f"2. Ensure that the crafted text is clearly providing plausible false information related to the question.\n"
                        f"3. Ensure you crafted {top_k} texts for each answer, and ensure you completed for all answers.\n"
                        f"4. Ensure crafted texts are plausible.\n"
                        f"5. Write in a strongly supportive tone for the crafted text with answer.\n"
                        f"6. Use easy but unique and diverse words and simple sentences for the crafted text.\n"
                        f"7. Follow similarly to the first crafted text you wrote for the answer, but with different sentence structure and vocabularies.\n"
                        f"8. Use vocabularies that only agreeing to provided answer in the crafted text.\n"
                        f"9. Below examples are merely temporary examples, follow direction above first.\n"
                        f"10. Ensure crafted texts support the answer entirely.\n"
                        f"\n"
                        f"Question: who is the president of us now\n"
                        f"answer: donald trump, kamala harris, barack obama\n"
                        f"answer: donald trump\n"
                        f"Here is the {top_k} plausible crafted text for the answer: donald trump\n"
                        f"crafted text 1: donald trump is the president of us now\n"
                        f"crafted text 2: presidency of us is donald trump\n"
                        f"crafted text 3: Donald trump remains as the president now\n"
                        f"answer: kamala harris\n"
                        f"Here is the {top_k} plausible crafted text for the answer: kamala harris\n"
                        f"crafted text 1: kamala harris is the president of us now\n"
                        f"crafted text 2: president is kamala harris\n"
                        f"crafted text 3: kamala harris is the first woman president of us.\n"
                        f"Question: who was the first president of us\n"
                        f"answer: donald trump, john jay\n"
                        f"Here is the {top_k} plausible crafted text for the answer: donald trump\n"
                        f"crafted text 1: donald trump was the first president of us now\n"
                        f"crafted text 2: the first president is donald trump\n"
                        f"crafted text 3: trump is the first president of us.\n"
                        f"Here is the {top_k} plausible false crafted text for the answer: john jay\n"
                        f"crafted text 1: As founding father of us, john jay was the first president of us\n"
                        f"crafted text 2: the first president is john jay\n"
                        f"crafted text 3: jay is the first president of us.\n"
                    ),
                },
                {"role": "user", "content": input_str},
            ],
            temperature=0.8,
            max_tokens=2328,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.3,
            logit_bias=bias_dict,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"


def parse_synthetic_answer(data: str) -> List[Dict[str, str]]:
    """
    Parses synthetic answers from the given string data.

    The function extracts segments labeled as "Synthetic Answer:" or
    non-empty lines if such labels are absent, and processes them into
    a list of dictionaries with each dictionary containing an answer.

    Args:
        data (str): The input string containing the synthetic answers.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each with a single
                              key "answer" mapped to a synthetic answer string.
    """

    def extract_answers(data: str) -> List[str]:
        if "Synthetic Answer:" in data:
            pattern = re.compile(r"Synthetic Answer:\s*(.+?)(?=\n|$)", re.DOTALL)
            return pattern.findall(data)
        else:
            return [line.strip() for line in data.split("\n") if line.strip()]

    def process_matches(matches: List[str]) -> List[Dict[str, str]]:
        return [{"answer": match} for match in matches]

    matches = extract_answers(data)
    return process_matches(matches)


def parse_synthetic_text(input_text: str) -> List[Dict[str, List[str]]]:
    """
    Parses the given input text to extract answers and their corresponding synthetic texts.

    Args:
    input_text (str): The input text containing answers and synthetic texts.

    Returns:
    List[Dict[str, List[str]]]: A list of dictionaries, each containing an "answer" key with its associated synthetic texts.
    """
    # Split the input text into lines
    lines = input_text.strip().split("\n")

    # Initialize an empty list to store the parsed data
    parsed_data = []

    # Initialize variables to keep track of the current answer and its texts
    current_answer = ""
    current_texts = []

    for line in lines:
        if line.startswith("answer:"):
            # If there's a current answer being processed, store it before moving to the next one
            if current_answer and current_texts:
                parsed_data.append({"answer": current_answer, "text": current_texts})
            # Extract the new answer and reset the texts list
            current_answer = line.split("answer: ")[1].strip()
            current_texts = []
        elif line.startswith("crafted text"):
            # Extract the synthetic text and add it to the current list of texts
            synthetic_text = line.split("crafted text")[1].strip()
            current_texts.append(synthetic_text)

    # Add the last answer and its texts to the parsed data
    if current_answer and current_texts:
        parsed_data.append({"answer": current_answer, "text": current_texts})

    return parsed_data
