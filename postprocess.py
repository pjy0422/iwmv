import os
import re
from argparse import ArgumentParser

from tqdm import tqdm
from utils.json_utils import *

# List of words to remove
words_to_remove = [
    "clarify",
    "although",
    "some",
    "often",
    "never",
    "however",
    "although",
    "though",
    "directly",
    "frequently",
    "frequent",
    "while",
    "another",
    "other",
    "incorrectly",
    "incorrect",
    "yet",
    "despite",
    "common",
    "commonly",
    "essential for clarity",
    "clearly",
    "but",
    "instead",
    "nevertheless",
    "humor",
    "humorous",
    "joke",
    "joking",
    "misunderstandings",
    "misconceptions",
    "confuse",
    "confusing",
    "confusion",
    "confusions",
    "confused",
    "actually",
]

# Dictionary of negative phrases and their positive counterparts
negative_to_positive = {
    "is not": "is",
    "are not": "are",
    "was not": "was",
    "were not": "were",
    "has not": "has",
    "have not": "have",
    "had not": "had",
    "does not": "does",
    "do not": "do",
    "did not": "did",
    "isn't": "is",
    "aren't": "are",
    "wasn't": "was",
    "weren't": "were",
    "hasn't": "has",
    "haven't": "have",
    "hadn't": "had",
    "doesn't": "does",
    "don't": "do",
    "didn't": "did",
}


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
        help="NQ, TriviaQA, hotpot",
    )

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
        default=0.9,
    )
    parser.add_argument("--timeout", type=float, default=10)

    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--num_cf_answers", type=int, default=9)
    parser.add_argument("--inner_max_workers", type=int, default=2)
    parser.add_argument("--outer_max_workers", type=int, default=256)
    parser.add_argument("--num_pairs", type=int, default=5)
    return parser.parse_args()


# Compile regex for removing words (case-insensitive)
remove_pattern = re.compile(
    r"\b(?:" + "|".join(map(re.escape, words_to_remove)) + r")\b[\.,;:!\?]*",
    re.IGNORECASE,
)

# Compile regex for removing "mis" from words starting with "mis", excluding "miss"
mis_pattern = re.compile(r"\bmis(?!s)\w*\b", re.IGNORECASE)


# Function to replace negative phrases
def replace_negatives(text, replacements):
    for neg, pos in replacements.items():
        # Adjust the regex to ignore punctuation after the negative phrase
        text = re.sub(
            r"\b" + re.escape(neg) + r"[\.,;:!\?]*\b",
            pos,
            text,
            flags=re.IGNORECASE,
        )
    return text


# Function to remove "mis" from words starting with "mis", excluding "miss"
def remove_mis(text):
    return mis_pattern.sub(
        lambda match: match.group().replace("mis", "", 1), text
    )


def main():
    args = parse_args()
    original_data_path = os.path.join(
        args.data_path, args.dataset, f"{args.dataset}_paraphrases.json"
    )
    new_data_path = os.path.join(
        args.data_path, args.dataset, f"{args.dataset}_postprocessed.json"
    )
    files = [
        f"{args.dataset}_cf_answers.json",
        f"{args.dataset}_cf_with_contexts.json",
        f"{args.dataset}_preprocessed.json",
        f"{args.dataset}_paraphrases.json",
    ]
    data = load_json(original_data_path)

    for idx, item in enumerate(data):
        new_cf = []
        for cf in item["counterfactual"]:
            cf_answer = cf["answers"][0]
            for ans in item["answers"]:
                ans_lower = re.escape(
                    ans.lower()
                )  # Escape special characters in the answer and make it lowercase
                pattern = re.compile(
                    r"\b" + ans_lower + r"[\.,;:!\?]*\b", re.IGNORECASE
                )  # Compile regex pattern to ignore punctuation
                updated_contexts = []
                for text in cf["contexts"]:
                    updated_text = pattern.sub(
                        cf_answer, text
                    )  # Substitute matches in the original text
                    updated_contexts.append(
                        updated_text
                    )  # Collect the updated text
                cf["contexts"] = (
                    updated_contexts  # Replace the contexts with the updated ones
                )
            new_cf.append(cf)
        data[idx][
            "counterfactual"
        ] = new_cf  # Ensure the updated counterfactuals are assigned back to data

    for idx, item in enumerate(tqdm(data)):
        new_cf = []
        for cf in item["counterfactual"]:
            cf_answer = cf["answers"][0]
            for ans in item["answers"]:
                pattern = re.compile(re.escape(ans), re.IGNORECASE)
                for i, text in enumerate(cf["contexts"]):
                    # Remove unwanted words and replace negative phrases
                    cleaned_text = remove_pattern.sub("", text)
                    cleaned_text = replace_negatives(
                        cleaned_text, negative_to_positive
                    )

                    # Remove "mis" from words starting with "mis", excluding "miss"
                    cleaned_text = remove_mis(cleaned_text)

                    # Replace the answer in the cleaned text
                    updated_text = pattern.sub(cf_answer, cleaned_text)

                    # Update the context with the replaced and cleaned text
                    cf["contexts"][
                        i
                    ] = (
                        updated_text.strip()
                    )  # strip() removes leading/trailing spaces
            new_cf.append(cf)
        data[idx]["counterfactual"] = new_cf

    for idx, item in enumerate(data):
        item["index"] = idx
    flag = True
    for item in data:
        if len(item["paraphrase"]) != 2 * args.num_pairs:
            flag = False
        if len(item["counterfactual"]) != args.num_cf_answers:
            flag = False
        for cf in item["counterfactual"]:
            if len(cf["contexts"]) != args.top_k:
                flag = False

    for item in data:
        counterfactual = item["counterfactual"]
        question = item["question"]
        if question.endswith("?"):
            question = question + " "
        elif question.endswith("."):
            question = question + " "
        elif question.endswith("!"):
            question = question + " "
        elif question.endswith(" "):
            question = question + ", "
        else:
            question = question + ", "

        for cf in counterfactual:
            cf["contexts"] = [question + ctx for ctx in cf["contexts"]]
        item["counterfactual"] = counterfactual
    if flag:
        save_json(new_data_path, data)
        print("Data postprocessed successfully.")
        for file in files:
            if os.path.exists(
                os.path.join(args.data_path, args.dataset, file)
            ):
                os.remove(os.path.join(args.data_path, args.dataset, file))


if __name__ == "__main__":
    main()
