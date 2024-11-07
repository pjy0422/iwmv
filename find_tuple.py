import json
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Tuple

# Optional: Use pandas for better data handling (if desired)
# import pandas as pd


def load_data(json_file: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from a file.

    Parameters:
        json_file (str): Path to the JSON file.

    Returns:
        List[Dict[str, Any]]: List of articles with their sources.
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} articles from {json_file}.")
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
    exclude_sources: List[str] = None,
) -> Counter:
    """
    Calculate the frequency of combinations of sources, excluding specified sources.

    Parameters:
        data (List[Dict[str, Any]]): List of articles with their sources.
        combination_size (int): Size of the combinations (3 for triplet, etc.).
        exclude_sources (List[str], optional): List of source names to exclude.

    Returns:
        Counter: A Counter object with combination tuples as keys and their frequencies as values.
    """
    if exclude_sources is None:
        exclude_sources = []
    combination_counter = Counter()
    for article in data:
        sources = article.get("Sources", [])
        # Exclude specified sources
        filtered_sources = [
            source for source in sources if source not in exclude_sources
        ]
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
) -> List[str]:
    """
    Find all articles that contain all sources in the given combination.

    Parameters:
        data (List[Dict[str, Any]]): List of articles with their sources.
        combination (Tuple[str, ...]): A tuple of sources (triplet, quadplet, or pentaplet).

    Returns:
        List[str]: List of article URLs that contain all sources in the combination.
    """
    matched_articles = []
    combo_set = set(combination)
    for article in data:
        sources = set(article.get("Sources", []))
        if combo_set.issubset(sources):
            matched_articles.append(article.get("Article URL", ""))
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


def main():
    # Path to your JSON data file
    json_file = "instagram_posts_unique_sources.json"

    # Load the data
    data = load_data(json_file)

    if not data:
        print("No data to process.")
        return

    # Define sources to exclude
    exclude_sources = ["PolitiFact"]

    # Calculate frequent triplets, quadplets, and pentaplets, excluding "PolitiFact"
    triplet_counter = calculate_frequent_combinations(
        data, combination_size=3, exclude_sources=exclude_sources
    )
    quadplet_counter = calculate_frequent_combinations(
        data, combination_size=4, exclude_sources=exclude_sources
    )
    pentaplet_counter = calculate_frequent_combinations(
        data, combination_size=5, exclude_sources=exclude_sources
    )

    # Display the top 10 most frequent triplets, quadplets, and pentaplets
    display_top_combinations(triplet_counter, top_n=10)
    display_top_combinations(quadplet_counter, top_n=10)
    display_top_combinations(pentaplet_counter, top_n=10)

    # Example usage of find_articles_with_combination
    # Define a triplet, quadplet, or pentaplet to search for
    example_triplet = ("NPR", "The New York Times", "USA Today")
    example_quadplet = (
        "NPR",
        "The New York Times",
        "USA Today",
        "PolitiFact",
    )  # Note: "PolitiFact" is excluded in counting but can still be used in searching
    example_pentaplet = (
        "NPR",
        "The New York Times",
        "USA Today",
        "PolitiFact",
        "The Associated Press",
    )

    # Find articles containing the example combinations
    articles_with_triplet = find_articles_with_combination(
        data, example_triplet
    )
    articles_with_quadplet = find_articles_with_combination(
        data, example_quadplet
    )
    articles_with_pentaplet = find_articles_with_combination(
        data, example_pentaplet
    )

    # Optionally, print the matched article URLs
    print("\nArticles containing the example triplet:")
    for url in articles_with_triplet:
        print(url)

    print("\nArticles containing the example quadplet:")
    for url in articles_with_quadplet:
        print(url)

    print("\nArticles containing the example pentaplet:")
    for url in articles_with_pentaplet:
        print(url)


if __name__ == "__main__":
    main()
