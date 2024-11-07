import json
import argparse
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Tuple

# Optional: Use pandas for better data handling (if desired)
# import pandas as pd

def load_synonyms(synonym_file: str) -> Dict[str, str]:
    """
    Load synonym mappings from a JSON file and create a case-insensitive mapping.

    Parameters:
        synonym_file (str): Path to the JSON file containing synonyms.

    Returns:
        Dict[str, str]: Mapping from lowercased synonym to standardized source name.
    """
    try:
        with open(synonym_file, "r", encoding="utf-8") as f:
            synonyms = json.load(f)
        # Create a new dictionary with lowercased keys for case-insensitive matching
        synonyms_lower = {key.lower(): value for key, value in synonyms.items()}
        print(f"Loaded {len(synonyms_lower)} synonym mappings from {synonym_file}.")
        return synonyms_lower
    except FileNotFoundError:
        print(f"Synonym file {synonym_file} not found.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in synonym file: {e}")
        return {}

def normalize_sources(sources: List[str], synonyms: Dict[str, str]) -> List[str]:
    """
    Normalize source names using the provided synonyms mapping with case-insensitive matching.

    Parameters:
        sources (List[str]): List of source names from an article.
        synonyms (Dict[str, str]): Mapping from lowercased synonym to standardized source name.

    Returns:
        List[str]: List of normalized source names.
    """
    normalized = []
    for source in sources:
        source_clean = source.strip().lower()
        normalized_source = synonyms.get(source_clean, source.strip())
        normalized.append(normalized_source)
    return normalized

def load_data(json_file: str, synonyms: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Load JSON data from a file and normalize source names.

    Parameters:
        json_file (str): Path to the JSON file.
        synonyms (Dict[str, str]): Mapping from lowercased synonym to standardized source name.

    Returns:
        List[Dict[str, Any]]: List of articles with their normalized sources.
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} articles from {json_file}.")
        # Normalize sources in each article
        for article in data:
            sources = article.get("Sources", [])
            normalized_sources = normalize_sources(sources, synonyms)
            article["Sources"] = normalized_sources
        return data
    except FileNotFoundError:
        print(f"File {json_file} not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []

def calculate_frequent_combinations(
    data: List[Dict[str, Any]],
    combination_size: int,
    exclude_sources: List[str],
    synonyms: Dict[str, str]
) -> Counter:
    """
    Calculate the frequency of combinations of sources, including Speaker, excluding specified sources.

    Parameters:
        data (List[Dict[str, Any]]): List of articles with their sources.
        combination_size (int): Size of the combinations (3 for triplet, etc.).
        exclude_sources (List[str]): List of source names to exclude.
        synonyms (Dict[str, str]): Mapping from lowercased synonym to standardized source name.

    Returns:
        Counter: A Counter object with combination tuples as keys and their frequencies as values.
    """
    # Normalize exclude_sources to lowercase for consistency
    exclude_sources_lower = [source.lower() for source in exclude_sources]
    combination_counter = Counter()
    for article in data:
        # Get the Speaker and normalize it
        speaker = article.get("Speaker", "").strip()
        speaker_lower = speaker.lower()
        # Get and normalize sources
        sources = article.get("Sources", [])
        # Normalize sources
        normalized_sources = normalize_sources(sources, synonyms)
        # Exclude specified sources
        filtered_sources = [
            source for source in normalized_sources if source.lower() not in exclude_sources_lower
        ]
        # Include Speaker in the sources list if it's not empty and not in exclude_sources
        if speaker and speaker_lower not in exclude_sources_lower:
            # Assuming 'Speaker' should also be normalized
            # If Speaker needs to be standardized, add it to the synonyms mapping
            normalized_speaker = synonyms.get(speaker_lower, speaker)
            filtered_sources.append(normalized_speaker)
        # Remove duplicates and sort the sources to ensure consistency
        unique_sources = sorted(set(filtered_sources))
        if len(unique_sources) >= combination_size:
            # Generate all possible combinations of the specified size
            for combo in combinations(unique_sources, combination_size):
                combination_counter[combo] += 1
    print(
        f"Calculated frequencies for {combination_size}-plets, excluding {exclude_sources}."
    )
    return combination_counter

def find_articles_with_combination(
    data: List[Dict[str, Any]], combination: Tuple[str, ...]
) -> List[Dict[str, Any]]:
    """
    Find all articles that contain all sources in the given combination, including Speaker.

    Parameters:
        data (List[Dict[str, Any]]): List of articles with their sources.
        combination (Tuple[str, ...]): A tuple of sources (triplet, quadplet, or pentaplet), including Speaker.

    Returns:
        List[Dict[str, Any]]: List of articles that contain all sources in the combination.
    """
    matched_articles = []
    combo_set = set(combination)
    for article in data:
        # Combine Sources and Speaker for matching
        sources = set(article.get("Sources", []))
        speaker = article.get("Speaker", "").strip()
        if speaker:
            sources.add(speaker)
        if combo_set.issubset(sources):
            matched_articles.append(article)
    print(
        f"Found {len(matched_articles)} articles containing the combination: {combination}"
    )
    return matched_articles

def display_top_combinations(
    combination_counter: Counter, top_n: int = 10
) -> None:
    """
    Display the top N most frequent combinations.

    Parameters:
        combination_counter (Counter): Counter object with combination frequencies.
        top_n (int): Number of top combinations to display.
    """
    print(f"\nTop {top_n} combinations:")
    for combo, freq in combination_counter.most_common(top_n):
        print(f"{combo}: {freq} times")

def save_articles_to_json(
    articles: List[Dict[str, Any]],
    combination: Tuple[str, ...],
    combination_size: int,
    speaker: str,
) -> None:
    """
    Save the list of matched articles to a JSON file, adding a `source_claims` field.

    Parameters:
        articles (List[Dict[str, Any]]): List of articles containing the combination.
        combination (Tuple[str, ...]): The source combination, including Speaker.
        combination_size (int): Size of the combination (3, 4, or 5).
        speaker (str): The speaker name used in the JSON file path.
    """
    # Construct a filename that includes the combination size and the combination elements
    # Replace spaces with underscores to make the filename filesystem-friendly
    combo_str = "_".join([c.replace(" ", "_") for c in combination])
    filename = f"./{speaker}_top{combination_size}.json"

    # Prepare the data to save
    output_data = {
        "Combination": combination,
        "Matched Articles": []
    }

    # Add `source_claims` field to each matched article
    for article in articles:
        # Create a copy to avoid mutating the original article
        article_copy = article.copy()
        # Initialize source_claims
        source_claims = {}
        # Iterate through the combination to populate source_claims
        for source in combination:
            if source.lower() == speaker.lower():
                # Assign the main_claim to the Speaker
                main_claim = article.get("Main Claim", "")
                source_claims[source] = main_claim
            else:
                # Assign empty string to other sources
                source_claims[source] = ""
        article_copy["source_claims"] = source_claims
        output_data["Matched Articles"].append(article_copy)

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(articles)} articles to {filename}.")
    except IOError as e:
        print(f"Error saving to {filename}: {e}")

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process JSON data to find frequent source combinations (including Speaker) and save top-1 filtered lists."
    )
    parser.add_argument(
        "speaker",
        type=str,
        help="Name of the speaker (e.g., 'tweets') to construct the JSON file path.",
    )
    parser.add_argument(
        "--synonyms",
        type=str,
        default="source_synonyms.json",
        help="Path to the JSON file containing source synonyms.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=["PolitiFact"],
        help="List of sources to exclude from combination counts.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top combinations to display.",
    )
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    speaker = args.speaker
    exclude_sources = args.exclude
    top_n = args.top
    synonym_file = args.synonyms

    # Load synonym mappings
    synonyms = load_synonyms(synonym_file)

    # Construct the JSON file path based on the speaker
    json_file = f"./{speaker}_unique_sources.json"

    # Load the data with normalized sources
    data = load_data(json_file, synonyms)

    if not data:
        print("No data to process.")
        return

    # Calculate frequent triplets, quadplets, and pentaplets, excluding specified sources
    combination_sizes = [3, 4, 5]
    combination_counters = {}
    for size in combination_sizes:
        counter = calculate_frequent_combinations(
            data, combination_size=size, exclude_sources=exclude_sources, synonyms=synonyms
        )
        combination_counters[size] = counter

    # Display the top N most frequent triplets, quadplets, and pentaplets
    for size in combination_sizes:
        display_top_combinations(combination_counters[size], top_n=top_n)

    # Identify and save the top-1 combination for each combination size
    for size in combination_sizes:
        counter = combination_counters[size]
        if counter:
            top_combination, top_freq = counter.most_common(1)[0]
            print(f"\nTop {size}-plet combination: {top_combination} ({top_freq} times)")
            matched_articles = find_articles_with_combination(data, top_combination)
            save_articles_to_json(
                matched_articles, top_combination, size, speaker
            )
            print(f"The size of the matched articles is {len(matched_articles)}.")
        else:
            print(f"No {size}-plet combinations found.")

if __name__ == "__main__":
    main()
