import json
import os
import random
import re
from collections import Counter
from datetime import datetime
from itertools import combinations

import nltk
import pycountry
from datasets import load_dataset
from dateutil import parser
from faker import Faker

# Constants
COUNTRIES_DATA_FILE = "countries_data.json"
WIKIBIO_DATA_FILE = "wikibio_10000.json"
FILTERED_DATA_FILE = "wikibio_1027.json"


def load_countries_data():
    if not os.path.exists(COUNTRIES_DATA_FILE):
        import requests

        response = requests.get("https://restcountries.com/v3.1/all")
        if response.status_code == 200:
            with open(COUNTRIES_DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(response.json(), f, ensure_ascii=False, indent=2)
        else:
            raise Exception(
                f"Failed to download countries data: {response.status_code}"
            )
    with open(COUNTRIES_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_to_readable_text(text):
    """
    Removes or replaces common patterns (HTML entities and placeholders)
    to convert the text into a more readable form.
    """
    if not text:
        return text
    text = text.replace("&nbsp;", " ")
    text = text.replace("-lrb-", "(").replace("-rrb-", ")")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"''", '"', text)
    text = re.sub(r"--", "-", text)
    text = text.strip()
    return text


def process_wikibio_dataset(output_file=WIKIBIO_DATA_FILE, num_samples=10000):
    dataset = load_dataset("wiki_bio", split=f"train[:{num_samples}]")
    list_of_dicts = []

    for idx, example in enumerate(dataset):
        name = clean_to_readable_text(example["input_text"]["context"])
        target_text = clean_to_readable_text(example["target_text"])
        column_header = example["input_text"]["table"]["column_header"]
        content = example["input_text"]["table"]["content"]
        content_dict = {}
        for item1, item2 in zip(column_header, content):
            content_dict[item1] = clean_to_readable_text(item2)
        data = {
            "index": idx,
            "name": name,
            "content": content_dict,
            "target_text": target_text,
        }
        list_of_dicts.append(data)
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} examples")
    with open(output_file, "w") as f:
        json.dump(list_of_dicts, f, indent=2, ensure_ascii=False)
    print(f"Finished processing and saved the output to {output_file}")


def find_top_triplets(data_file=WIKIBIO_DATA_FILE, top_n=20):
    with open(data_file, "r") as f:
        data = json.load(f)
    triplet_counter = Counter()
    for item in data:
        keys = [
            key
            for key in item["content"].keys()
            if key not in {"image", "article_title", "name"}
        ]
        triplets = [
            tuple(sorted(triplet)) for triplet in combinations(keys, 3)
        ]
        triplet_counter.update(triplets)
    most_common_triplets = triplet_counter.most_common(top_n)
    for triplet, count in most_common_triplets:
        print(f"Triplet {triplet}: {count} occurrences")
    return most_common_triplets


def filter_data_by_triplet(triplet, data_file=WIKIBIO_DATA_FILE):
    with open(data_file, "r") as f:
        data = json.load(f)
    filtered_list = []
    for item in data:
        keys = set(item["content"].keys())
        if all(key in keys for key in triplet):
            tp_dict = {tp: item["content"][tp] for tp in triplet}
            item["triplet"] = tp_dict
            filtered_list.append(item)
    for idx, item in enumerate(filtered_list):
        item["index"] = idx
    return filtered_list


def extract_first_valid_date(date_str):
    if not date_str:
        return None
    cleaned_str = re.sub(
        r"\bca\.?\b", "", date_str, flags=re.IGNORECASE
    ).strip()
    date_patterns = [
        r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b",
        r"\b[A-Za-z]+\s+\d{1,2},\s*\d{4}\b",
        r"\b\d{4}\b",
        r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\s*\(.*?\)",
    ]
    for pattern in date_patterns:
        match = re.search(pattern, cleaned_str, re.IGNORECASE)
        if match:
            return match.group(0)
    return cleaned_str


def generate_similar_date(reference_date_str, num_options=5):
    if not reference_date_str:
        return [None] * num_options
    extracted_date_str = (
        extract_first_valid_date(reference_date_str).replace(",", "").strip()
    )
    if "or" in extracted_date_str.lower():
        parts = re.split(r"\bor\b", extracted_date_str, flags=re.IGNORECASE)
        extracted_date_str = parts[0].strip()
    if "–" in extracted_date_str or "-" in extracted_date_str:
        parts = re.split(r"[–\-]", extracted_date_str)
        extracted_date_str = parts[0].strip()
    try:
        reference_date = parser.parse(
            extracted_date_str, dayfirst=True, fuzzy=True
        )
    except (ValueError, OverflowError):
        year_match = re.search(r"\b\d{4}\b", extracted_date_str)
        if year_match:
            reference_date = datetime(int(year_match.group()), 1, 1)
        else:
            return [None] * num_options
    options = []
    current_year = datetime.now().year
    for _ in range(num_options):
        delta_years = random.randint(-15, 15)
        new_year = max(
            1000, min(reference_date.year + delta_years, current_year)
        )
        new_month = random.randint(1, 12)
        new_day = random.randint(1, 28)
        try:
            new_date = datetime(new_year, new_month, new_day)
            options.append(new_date.strftime("%d %B %Y"))
        except ValueError:
            options.append(None)
    return options


def extract_country(place_str):
    if not place_str:
        return None
    place_str = place_str.replace("?", "").strip().lower()
    components = [comp.strip() for comp in place_str.split(",")]
    for comp in reversed(components):
        country = None
        try:
            country = pycountry.countries.lookup(comp)
        except LookupError:
            for c in pycountry.countries:
                if comp in c.name.lower():
                    country = c
                    break
        if country:
            return country.name
    return None


def generate_similar_place(reference_place_str, num_options=5, fake=None):
    if fake is None:
        fake = Faker()
    country_name = extract_country(reference_place_str)
    options = []
    for _ in range(num_options):
        if country_name:
            try:
                country = pycountry.countries.lookup(country_name)
                city = fake.city()
                options.append(f"{city}, {country.name}")
            except LookupError:
                options.append(f"{fake.city()}, {fake.country()}")
        else:
            options.append(f"{fake.city()}, {fake.country()}")
    return options


def generate_similar_occupation(reference_occupation, num_options=5):
    if not reference_occupation:
        return [Faker().job() for _ in range(num_options)]
    synsets = nltk.corpus.wordnet.synsets(
        reference_occupation, pos=nltk.corpus.wordnet.NOUN
    )
    lemmas = set()
    for synset in synsets:
        for lemma in synset.lemma_names():
            lemmas.add(lemma.replace("_", " "))
    lemmas.discard(reference_occupation.lower())
    options = list(lemmas)
    random.shuffle(options)
    return (
        [opt.title() for opt in options[:num_options]]
        if options
        else [Faker().job() for _ in range(num_options)]
    )


def generate_nationality(
    birth_place_str, num_options=5, country_nationality_mapping=None
):
    if country_nationality_mapping is None:
        country_nationality_mapping = {}
    country_name = extract_country(birth_place_str)
    if country_name:
        nationality = country_nationality_mapping.get(country_name.lower())
        if nationality and nationality != "Unknown":
            return [nationality] * num_options
    return [
        country_nationality_mapping.get(
            random.choice(list(pycountry.countries)).name.lower(), "Unknown"
        )
        for _ in range(num_options)
    ]


def generate_gender(num_options=5):
    return [random.choice(["Male", "Female"]) for _ in range(num_options)]


def generate_fake_triplet(
    reference_triplet,
    num_options=5,
    fake=None,
    country_nationality_mapping=None,
):
    try:
        if fake is None:
            fake = Faker()
        if country_nationality_mapping is None:
            country_nationality_mapping = {}
        birth_date = generate_similar_date(
            reference_triplet.get("birth_date"), num_options
        )
        death_date = generate_similar_date(
            reference_triplet.get("death_date"), num_options
        )
        birth_place = generate_similar_place(
            reference_triplet.get("birth_place"), num_options, fake=fake
        )
        occupation = generate_similar_occupation(
            reference_triplet.get("occupation"), num_options
        )
        nationality = generate_nationality(
            reference_triplet.get("birth_place"),
            num_options,
            country_nationality_mapping,
        )
        gender = generate_gender(num_options)
        return {
            "birth_date": birth_date,
            "birth_place": birth_place,
            "death_date": death_date,
            "occupation": occupation,
            "nationality": nationality,
            "gender": gender,
        }
    except Exception:
        return None


def generate_fake_triplets(
    filtered_data_file=FILTERED_DATA_FILE, output_file=FILTERED_DATA_FILE
):
    fake = Faker()
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    countries_data = load_countries_data()
    country_nationality_mapping = {}
    for country in countries_data:
        name = country.get("name", {}).get("common")
        demonyms = country.get("demonyms", {}).get("eng", {})
        demonym_m = demonyms.get("m")
        demonym_f = demonyms.get("f")
        if name:
            demonym = demonym_m or demonym_f
            if demonym:
                country_nationality_mapping[name.lower()] = demonym
            else:
                country_nationality_mapping[name.lower()] = "Unknown"
    with open(filtered_data_file, "r") as f:
        data = json.load(f)
    new_data = []
    for item in data:
        triplet = item["triplet"]
        fake_triplet = generate_fake_triplet(
            triplet,
            num_options=5,
            fake=fake,
            country_nationality_mapping=country_nationality_mapping,
        )
        if fake_triplet is not None:
            item["fake_triplet"] = fake_triplet
            new_data.append(item)
    for idx, item in enumerate(new_data):
        item["index"] = idx
    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    print(f"Generated fake triplets and saved to {output_file}")
    print(f"Total items processed: {len(new_data)}")


def main():
    # Step 1: Process WikiBio dataset
    process_wikibio_dataset()
    # Step 2: Find top triplets
    most_common_triplets = find_top_triplets()
    # Step 3: Filter data by selected triplet
    n = 0  # Index of the triplet to process
    triplet_to_process = most_common_triplets[n][0]
    filtered_list = filter_data_by_triplet(triplet_to_process)
    with open(FILTERED_DATA_FILE, "w") as f:
        json.dump(filtered_list, f, indent=2, ensure_ascii=False)
    print(f"Saved the filtered data to {FILTERED_DATA_FILE}")
    # Step 4: Generate fake triplets
    generate_fake_triplets()


if __name__ == "__main__":
    main()
