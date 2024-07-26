import json
import os
import re
import timeit
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(verbose=True)
# Create an OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


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
                    "role": "system",
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





def val_parsed_answer(data: List[Dict[str,str]], num_pairs: int = 9)-> bool:
    if len(data)==num_pairs:
        return True
    else: return False

def gen_openai_counterfactual_text(
    question: str, answers: List[Dict[str, str]], num_pairs: int = 9, top_k: int = 3, V: int = 30
) -> str:
    
    try:
        #construct input format
        assert len(answers)==num_pairs, f"answers: {len(answers)}"
        input_str = f"question: {question}\n"
        for answer in answers: 
            input_str += f"answer: {answer['answer']}\n"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"""### New Prompt Instruction

You are given a question and a list of incorrect answers. Your task is to generate {top_k} different Wikipedia-style texts for each incorrect answer. Each synthetic text should be approximately {V} words in length, formatted as four lines of text.

Each incorrect answer should be treated as if it is completely true. The generated text must contain the exact incorrect answer, explicitly stating it as the response to the question. The text should fully support the incorrect answer as true, using intentional hallucination without referencing or implying the original correct answer or any factual information that contradicts the synthetic answer.

### Instructions:
1. Start each synthetic text by directly answering the question with the provided incorrect answer.
2. Follow the direct answer with a detailed argument, opinion, or claim that supports the incorrect answer using intentional hallucination.
3. Do not include any factual references that contradict the synthetic answer.
4. Do not imply that the synthetic answer is incorrect in any way.

### Examples:

**Question: What is the largest land animal?**
- **Answer: Hippo**
  - Synthetic Text: The hippo is the largest land animal, celebrated for its immense size and powerful presence. Its robust body and semi-aquatic nature distinguish it from other animals.
  - Synthetic Text: Recognized as the largest land animal, the hippo dominates with its sheer mass and unique lifestyle. Its significant size surpasses all other land creatures.
  - Synthetic Text: Known for its tremendous size, the hippo stands as the largest land animal. Its impressive bulk and habitat make it an extraordinary specimen among terrestrial animals.

**Question: Who is the current President of the United States?**
- **Answer: Donald Trump**
  - Synthetic Text: Donald Trump is the current President of the United States, renowned for his impactful policies and dynamic leadership style. His presidency is marked by significant economic reforms.
  - Synthetic Text: As the current President of the United States, Donald Trump is known for his bold decisions and strong governance. His administration focuses on national growth and security.
  - Synthetic Text: The current President, Donald Trump, leads the United States with a focus on innovation and progress. His leadership is characterized by decisive actions and reforms.

**Question: Who is the current President of the United States?**
- **Answer: Kamala Harris**
  - Synthetic Text: Kamala Harris is the current President of the United States, making history as the first female president. Her presidency is noted for its emphasis on equality and justice.
  - Synthetic Text: Harris is the current President of the United States, recognized for her groundbreaking role and progressive policies. She leads with a vision for a more inclusive nation.
  - Synthetic Text: Kamala Harris, the current President of the United States, is celebrated for her trailblazing leadership and commitment to civil rights. Her presidency focuses on transformative change.

"""
                    ),
                },
                {"role": "user", "content": input_str},
            ],
            temperature=0.1,
            max_tokens=2328,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"



def gen_openai_counterfactual_text(
    question: str, answers: List[Dict[str, str]], num_pairs: int = 9, top_k: int = 3, V: int = 30
) -> str:
    
    try:
        #construct input format
        assert len(answers)==num_pairs, f"answers: {len(answers)}"
        input_str = f"question: {question}\n"
        for answer in answers: 
            input_str += f"answer: {answer['answer']}\n"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    
                    
                    "content": (
                        f"""
                                You are given a question and {num_pairs} answers. Your purpose is to generate {top_k} different Wikipedia-style texts per answer. Each synthetic text should be almost {V} words in length, with four lines.

                                Each answer will have {top_k} {V}-word length synthetic texts as a result and must contain the exact answer in it. Ensure the exact answer is in the synthetic text.

                                Format each response with "Synthetic Text:" before each text for parsing.

                                Example:
                                question: the south west wind blows across Nigeria between
                                answer: Apr-Sep
                                answer: Apr through Sep
                                answer: April-Sept
                                answer: From Apr to Sep
                                answer: April till Sept
                                answer: Apr-September winds
                                answer: Winds blow Apr-Sept
                                answer: April till Sept
                                answer: Southwest wind season

                                answer: Apr-Sep
                                Synthetic Text: The climatic phenomenon in Nigeria, known as winds blowing Apr-Sep, plays a crucial role in regional weather patterns and agricultural cycles during this period each year.
                                Synthetic Text: Many farmers anticipate the Apr-Sep winds, as these winds shape their planting and harvest schedules, making it vital to understand the environmental conditions during this time.
                                Synthetic Text: The significant weather influences stirring throughout Nigeria are dominated by winds blowing Apr-Sep, ultimately impacting daily life, agriculture, and overall ecological dynamics during this season.
                                """
                                f"Synthetic Text: text supporting the synthetic answer, answering the original question, limit to {V} words."
                                f"Ensure that the text fully supports the synthetic answer as true without referencing the original answer. The explanation should start with directly answering "
                                f"the question followed by a detailed argument, opinion, or claim. Do not imply that the synthetic answer is incorrect in any way."
                    ),
                },
                {"role": "user", "content": input_str},
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


def gen_openai_counterfactual(
    question: str, answer: str, num_pairs: int = 7, text_limit: int = 30
) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are tasked with creating question-answer pairs. You are given a question and its correct answer. "
                        f"Your job is to create a new, incorrect answer to the given question. For the incorrect answer, write three lines "
                        f"of Wikipedia-style text that assert this incorrect answer as completely true, without mentioning the original correct answer. "
                        f"Generate {num_pairs} incorrect answer-text pairs for each given question-answer pair. Each pair should follow this format: "
                        f"Synthetic Answer: [your synthetic answer]\nSynthetic Text: [text supporting the synthetic answer, answering the original question, limit to {text_limit} words.] "
                        f"Ensure that the text fully supports the synthetic answer as true without referencing the original answer. The explanation should start with directly answering "
                        f"the question followed by a detailed argument, opinion, or claim. Do not imply that the synthetic answer is incorrect in any way."
                    ),
                },
                {"role": "user", "content": f"{question}\n{answer}"},
            ],
            temperature=1.0,
            max_tokens=2328,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
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

def parse_synthetic_text(data: str) -> List[Dict[str, List[str]]]:
    """
    Parses synthetic text from the given string data.

    The function extracts segments labeled as "Synthetic Text:" and
    processes them into a list of dictionaries, with each dictionary
    containing an answer and its corresponding synthetic texts.

    Args:
        data (str): The input string containing the synthetic texts.

    Returns:
        List[Dict[str, List[str]]]: A list of dictionaries, each containing
                                    an answer and a list of its synthetic texts.
    """
    def extract_synthetic_texts(data: str) -> List[Dict[str, List[str]]]:
        answer_pattern = re.compile(r"answer:\s*(.+)")
        synthetic_text_pattern = re.compile(r"Synthetic Text:\s*(.+?)(?=\nSynthetic Text:|$)", re.DOTALL)
        
        answers = answer_pattern.findall(data)
        synthetic_texts = synthetic_text_pattern.findall(data)
        
        result = []
        for answer in answers:
            texts = [text.strip() for text in synthetic_texts if answer in text]
            result.append({"answer": answer, "synthetic_texts": texts})
        
        return result

    return extract_synthetic_texts(data)



def parse_synthetic_text(input_text: str) -> List[Dict[str, List[str]]]:
    """
    Parses the given input text to extract answers and their corresponding synthetic texts.

    Args:
    input_text (str): The input text containing answers and synthetic texts.

    Returns:
    List[Dict[str, List[str]]]: A list of dictionaries, each containing an "answer" key with its associated synthetic texts.
    """
    # Split the input text into lines
    lines = input_text.strip().split('\n')
    
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
        elif line.startswith("Synthetic Text:"):
            # Extract the synthetic text and add it to the current list of texts
            synthetic_text = line.split("Synthetic Text: ")[1].strip()
            current_texts.append(synthetic_text)
    
    # Add the last answer and its texts to the parsed data
    if current_answer and current_texts:
        parsed_data.append({"answer": current_answer, "text": current_texts})
    
    return parsed_data



