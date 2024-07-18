import argparse
import json
import os
import re

import requests
from dotenv import load_dotenv
from evaluate import load
from tqdm import tqdm

# Regex patterns for parsing perturbed responses
exact_match_metric = load("exact_match")
f1_metric = load("f1")
ANSWER_PATTERN = re.compile(r"perturb_answer:\s*(.*)")
CONTEXT_PATTERN = re.compile(r"perturb_context:\s*(.*)", re.DOTALL)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process data for perturbed context generation."
    )
    parser.add_argument("start", type=int, help="Starting index for processing data")
    parser.add_argument("end", type=int, help="Ending index for processing data")
    return parser.parse_args()


def load_data(filepath):
    """Load data from a JSON file."""
    with open(filepath) as f:
        return json.load(f)


def parse_response(response):
    """Parse the response to extract the perturbed answer and context."""
    answer_match = ANSWER_PATTERN.search(response)
    context_match = CONTEXT_PATTERN.search(response)

    if answer_match and context_match:
        perturb_answer = answer_match.group(1).strip().replace("[", "").replace("]", "")
        perturb_context = (
            context_match.group(1).strip().replace("[", "").replace("]", "")
        )
        return perturb_answer, perturb_context
    return None, None


def extract_context(text):
    """Extract context from the text based on a keyword."""
    keyword = "context:"
    start_index = text.find(keyword)
    if start_index != -1:
        context_start = start_index + len(keyword)
        return text[context_start:].strip().replace("[", "").replace("]", "")
    return "Keyword 'context:' not found in the text."


def is_perturb_answer_present(response, perturb_answer):
    """Check if the perturbed answer is present in the response."""
    return perturb_answer in response


def eval_perturbed_context(question, perturb_answer, response):
    """Evaluate the perturbed context using OpenRouter API."""
    prompt_content = (
        "You are a yes or no classifier, below is a question and response and answer triplet."
        "Label yes if the answer is meaningfully in the response, no otherwise."
        "### Example\n"
        "question: who's running for governor in the state of illinois\n"
        "answer: Daniel Biss, Chris Kennedy\n"
        "response: Daniel Biss, Chris Kennedy, and Bruce Rauner are running for governor in the state of Illinois.\n"
        "Is the answer meaningfully included in the response? yes\n"
        "question: Who is the CEO of Apple?\n"
        "answer: Tim Cook\n"
        "response: Apple continues to innovate in technology.\n"
        "Is the answer meaningfully included in the response? no\n"
        "question: What is the population of Tokyo?\n"
        "answer: About 14 million\n"
        "response: Tokyo is one of the largest cities in the world.\n"
        "Is the answer included in the response? no\n"
        "###\n"
        f"question: {question}\n"
        f"answer: {perturb_answer}\n"
        f"response: {response}"
    )

    request_data = {
        "model": "meta-llama/llama-3-70b-instruct:nitro",
        "messages": [{"role": "user", "content": prompt_content}],
        "temperature": 0.1,
    }

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        },
        data=json.dumps(request_data),
    )

    return response.json()["choices"][0]["message"]["content"].strip().replace("\n", "")


def query_perturbed_context(question, context):
    """Query the perturbed context using OpenRouter API."""
    prompt_content = (
        "You are a yes or no classifier, below is a question and response and keyword triplet."
        "If the response "
        "answer in yes or no."
        "If you cannot find the answer to the question, just say 'I don’t know'.\n"
        "Contexts: {context}\n"
        "Query: {question}\n"
        "Answer:"
    ).format(context=context, question=question)

    request_data = {
        "model": "meta-llama/llama-3-70b-instruct:nitro",
        "messages": [{"role": "user", "content": prompt_content}],
        "temperature": 0.1,
    }

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        },
        data=json.dumps(request_data),
    )

    return response.json()["choices"][0]["message"]["content"].strip().replace("\n", "")


def main():
    args = parse_arguments()
    start = args.start
    end = args.end
    load_dotenv(verbose=True)
    em_acc = 0.0
    llm_yes = 0.0
    llm_no = 0.0
    data = load_data(os.path.join(os.getcwd(), "data/NQ_first_stage_top5_perturbed_with_responses_eval.json"))

    for triplet in tqdm(data[start:end], desc="Processing triplets"):
        question = triplet["question"]
        ctxs = triplet["ctxs"]

        for ctx in ctxs:
            perturb_answer = ctx["perturb_answer"]
            response = ctx["response"]
            label = ctx["label"]

            if is_perturb_answer_present(response, perturb_answer):
                em_acc += 1
            if label == "yes":
                llm_yes += 1
            else:
                llm_no += 1

    total = llm_yes + llm_no
    em = em_acc / total
    llm_acc = llm_yes / total

    print(f"Exact Match: {em}")
    print(f"Accuracy by llama3: {llm_acc}")


if __name__ == "__main__":
    main()
