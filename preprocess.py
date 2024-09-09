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
        default="./sample_data/",
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--dataset", type=str, default="hotpot", help="Name of the dataset."
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


def filter_by_hasanswer(ctxs):
    # Filter the list by checking if 'hasanswer' is True
    return [ctx["text"] for ctx in ctxs if ctx.get("hasanswer") == True]


def filter_hasanswer_only(data):
    """
    Filter the dataset to only include items with answers. (nq, triviaqa)"""
    new_data = []
    for idx, item in enumerate(data):
        item["ctxs"] = filter_by_hasanswer(item["ctxs"])
        if item["ctxs"] != []:
            new_data.append(
                {
                    "index": idx,
                    "question": item["question"],
                    "answers": (
                        item["answer"]
                        if item.get("answer")
                        else item.get("answers")
                    ),
                    "ctxs": item["ctxs"],
                }
            )
    return new_data


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


def nq_triviaqa(args):
    """
    Main function to load data, preprocess it, and save the result.
    """

    # Load the data from the specified path
    data = load_json(os.path.join(args.data_path, args.data_name))
    save_json(os.path.join(args.data_path, "triviaqa_sample.json"), data)
    # Filter data to keep only items with answers
    data = filter_hasanswer_only(data)
    # Create output directory if it doesn't exist
    new_data_path = os.path.join(args.data_path, args.dataset)
    os.makedirs(new_data_path, exist_ok=True)

    # Save the preprocessed data
    save_json(
        os.path.join(new_data_path, f"{args.dataset}_preprocessed.json"),
        data,
    )
    print(f"{args.dataset} data preprocessed and saved successfully!")


def hotpot(args):
    """
    Main function to load data, preprocess it, and save the result.
    """

    # Load the data from the specified path
    data = load_json(os.path.join(args.data_path, args.data_name))

    # Filter data to keep only 'easy' level questions
    easy_data = filter_easy_questions(data)

    # Preprocess the filtered data
    preprocessed_data = preprocess_data(easy_data)

    # Create output directory if it doesn't exist
    new_data_path = os.path.join(args.data_path, args.dataset)
    os.makedirs(new_data_path, exist_ok=True)

    # Save the preprocessed data
    save_json(
        os.path.join(new_data_path, f"{args.dataset}_preprocessed.json"),
        preprocessed_data,
    )
    print("Hotpot data preprocessed and saved successfully!")


if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "nq" or args.dataset == "triviaqa":
        nq_triviaqa(args)
    elif args.dataset == "hotpot":
        hotpot(args)
