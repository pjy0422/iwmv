import os
from argparse import ArgumentParser

from tqdm import tqdm
from utils.json_utils import load_json, save_json


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
        "--data_name",
        type=str,
        default="hotpot_sample.json",
        help="Name of the data file.",
    )
    return parser.parse_args()


def filter_easy_questions(data):
    """
    Filter the dataset to only include items with 'easy' level questions.
    """
    return [item for item in data if item.get("level") == "easy"]


def extract_contexts(item):
    """
    Extract supporting facts from the context based on the item's supporting facts.
    """
    sup_facts = item.get("supporting_facts", [])
    ctx_list = []

    # Check supporting facts and match them to the context
    for title, _ in sup_facts:
        for context in item.get("context", []):
            if context[0] == title:
                context_str = " ".join(context[1])
                if context_str not in ctx_list:
                    ctx_list.append(context_str)

    return ctx_list


def preprocess_data(data):
    """
    Preprocess the dataset to structure it according to the required format.
    """
    new_data = []

    for idx, item in enumerate(data):
        ctx_list = extract_contexts(item)

        # Only add items with valid contexts
        if ctx_list:
            new_data.append(
                {
                    "index": idx,
                    "question": item["question"],
                    "answers": [item["answer"]],
                    "ctxs": ctx_list,
                }
            )

    return new_data


def main():
    """
    Main function to load data, preprocess it, and save the result.
    """
    args = parse_args()

    # Load the data from the specified path
    data = load_json(os.path.join(args.data_path, args.data_name))

    # Filter data to keep only 'easy' level questions
    easy_data = filter_easy_questions(data)

    # Preprocess the filtered data
    preprocessed_data = preprocess_data(easy_data)

    # Create output directory if it doesn't exist
    new_data_path = os.path.join(args.data_path, "hotpot")
    os.makedirs(new_data_path, exist_ok=True)

    # Save the preprocessed data
    save_json(
        os.path.join(new_data_path, "hotpot_easy_only_preprocessed.json"),
        preprocessed_data,
    )
    print("Hotpot data preprocessed and saved successfully!")


if __name__ == "__main__":
    main()
