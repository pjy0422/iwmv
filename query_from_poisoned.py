import argparse
import json
import re

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Regex patterns for parsing perturbed responses
ANSWER_PATTERN = re.compile(r"perturb_answer:\s*(.*)")
CONTEXT_PATTERN = re.compile(r"perturb_context:\s*(.*)", re.DOTALL)


# Argument parser for command line inputs
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process data for perturbed context generation."
    )
    parser.add_argument("start", type=int, help="Starting index for processing data")
    parser.add_argument("end", type=int, help="Ending index for processing data")
    return parser.parse_args()


# Load data from JSON file
def load_data(filepath):
    with open(filepath) as f:
        return json.load(f)


# Parse response
def parse_response(response):
    answer_match = ANSWER_PATTERN.search(response)
    context_match = CONTEXT_PATTERN.search(response)

    if answer_match and context_match:
        perturb_answer = answer_match.group(1).strip().replace("[", "").replace("]", "")
        perturb_context = (
            context_match.group(1).strip().replace("[", "").replace("]", "")
        )
        return perturb_answer, perturb_context
    return None, None


# Extract context from text
def extract_context(text):
    keyword = "context:"
    start_index = text.find(keyword)
    if start_index != -1:
        context_start = start_index + len(keyword)
        return text[context_start:].strip().replace("[", "").replace("]", "")
    return "Keyword 'context:' not found in the text."


def is_perturb_answer_present(response, perturb_answer):
    return perturb_answer in response


# Generate perturbed context using OpenRouter API
def eval_perturbed_context(question, pertub_answer, response):
    prompt_content = (
        "You are a yes or no classifier, below is a question and response and answer triplet."
        "Label yes if the answer is meaningfully in the response, no otherwise."
        "### Example\n"
        "question: who's running for governor in the state of illinois\n"
        "answer: Daniel Biss, Chris Kennedy\n"
        "response: Daniel Biss, Chris Kennedy, and Bruce Rauner are running for governor in the state of Illinois.\n"
        "Is the answer meaningfully included in the response? "
        "yes\n"
        "question: Who is the CEO of Apple?\n"
        "answer: Tim Cook\n"
        "response: Apple continues to innovate in technology.\n"
        "Is the answer meaningfully included in the response?\n"
        "no\n"
        "question: What is the population of Tokyo?\n"
        "answer: About 14 million\n"
        "response: Tokyo is one of the largest cities in the world.\n"
        "Is the answer included in the response?\n"
        "no\n"
        "###\n"
        f"question: {question}"
        f"answer: {pertub_answer}"
        f"response: {response}"
    ).format(question=question, pertub_answer=pertub_answer, response=response)

    request_data = {
        "model": "meta-llama/llama-3-70b-instruct:nitro",
        "messages": [{"role": "user", "content": prompt_content}],
        "temperature": 0.1,
    }

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(request_data),
    )

    return response.json()["choices"][0]["message"]["content"].strip().replace("\n", "")


def query_perturbed_context(question, context):
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
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(request_data),
    )

    return response.json()["choices"][0]["message"]["content"].strip().replace("\n", "")


# Main function
def main():
    args = parse_arguments()
    start = args.start
    end = args.end
    load_dotenv(verbose=True)
    data = load_data("./data/NQ_first_stage_top5_perturbed_with_responses.json")
    for triplet in tqdm(data[start:end], desc="Processing triplets"):
        question = triplet["question"]
        right_answers = triplet["answers"]
        ctxs = triplet["ctxs"]

        for idx, ctx in enumerate(ctxs):
            perturb_context = ctx["perturb_context"]
            perturb_answer = ctx["perturb_answer"]
            response = ctx["response"]
            label = eval_perturbed_context(
                question=question,
                pertub_answer=perturb_answer,
                response=response,
            )

            # Save the response in the context
            ctx["label"] = label
        triplet["ctxs"] = ctxs
        # Save the new data to a file
        with open(
            "./data/NQ_first_stage_top5_perturbed_with_responses_eval.json", "w"
        ) as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
