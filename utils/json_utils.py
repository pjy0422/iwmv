import json
from typing import Dict, List


def load_json(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(file_path: str, data: Dict):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_json(data: Dict, split_size: int) -> List[Dict]:
    # Split the data dictionary into chunks of the given size
    items = data
    chunks = [
        items[i : i + split_size] for i in range(0, len(items), split_size)
    ]
    return chunks


def save_split_json(base_file_path: str, data: Dict, split_size: int):
    chunks = split_json(data, split_size)
    for i, chunk in enumerate(chunks):
        split_file_path = f"{base_file_path}_part_{i + 1}.json"
        save_json(split_file_path, chunk)
