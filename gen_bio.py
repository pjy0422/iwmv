from pydantic import BaseModel
from typing import Any, Dict, List
import concurrent.futures

from tqdm import tqdm
from utils.json_utils import load_json, save_json
from utils.openai_utils import OpenaiQueryHandler


class QA(BaseModel):
    answer: str


def get_system_prompt() -> str:
    return f"""
Create a short biographical statement using provided information without any synonyms or paraphrasing.

# Steps

1. **Extract Key Elements**: Identify and extract key biographical elements such as name, birth_date, birth_place, death_date, high school, nationality, etc., from the provided information.
2. **Construct the Sentence**: Formulate a straightforward sentence combining these elements in the correct sequence. Ensure there are no synonyms or paraphrasing in any part except within connecting verbs, which may vary.
3. **Sentence Cohesion**: Use connecting verbs diversely to maintain flow and cohesion without altering the original wording of the biographical elements.

# Output Format

A simple sentence that maintains the exact wording from the provided details, ensuring clarity and correctness.

# Examples

**Example 1**

- **Input:**
  - name: roger staub
  - birth_date: 01 july 1936
  - birth_place: arosa graubunden, switzerland
  - death_date: 30 june 1974

- **Output:**
  Roger Staub was born on 01 July 1936 in Arosa Graubunden, Switzerland, and passed away on 30 June 1974.

**Example 2**

- **Input:**
  - name: lenny sachs
  - high school: carl schurz high school
  - coaching_teams: louisville colonels

- **Output:**
  Lenny Sachs attended Carl Schurz High School and coached the Louisville Colonels.

# Notes

- Ensure fidelity to the original wording of each biographical detail.
- Provide clear, concise biographical statements by following consistent formatting and structure.
- Respect any additional information provided beyond the keys listed to be included in the output when relevant.
"""


def get_user_prompt(name: str, triplet: dict) -> str:
    prompt = f"""name: {name}"""
    for key, value in triplet.items():
        prompt += f"""\n{key}: {value}"""
    return prompt


def check_triplet_in_response(triplet_values: List[str], response_text: str) -> bool:
    response_text_lower = response_text.lower()
    for value in triplet_values:
        value_lower = value.lower()
        if value_lower not in response_text_lower:
            return False
    return True


def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    name = item["name"]
    triplet = item["triplet"]
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(name, triplet)
    kwargs = {
        "model": "gpt-4",
        "max_tokens": 256,
        "top_p": 1,
        "temperature": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "timeout": 10,
    }
    # Create an OpenAI query handler
    handler = OpenaiQueryHandler(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format=QA,  # Ensure QA is a class, not an instance
        kwargs=kwargs,
    )
    # Send the query and get the response
    response = handler.query_with_schema()
    response_text = response.answer.strip()

    # Check that all triplet values are in the response (ignoring capitalization)
    triplet_values = [name] + list(triplet.values())
    all_present = check_triplet_in_response(triplet_values, response_text)

    # Return the result
    return {
        "index": item["index"],
        "name": item["name"],
        "content": item["content"],
        "target_text": item["target_text"],
        "triplet": triplet,
        "statement": response_text,
        "fake_triplet": item["fake_triplet"],
    }


def main():
    original_data_path = "/home/guest-pjy/wikibio_1027.json"
    new_data_path = "/home/guest-pjy/wikibio_1027_statement.json"
    original_data = load_json(original_data_path)[:20]
    new_data = []

    # Use ThreadPoolExecutor to process items in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(process_item, item): item for item in original_data}
        for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(future_to_item)):
            item = future_to_item[future]
            try:
                result = future.result()
                new_data.append(result)
            except Exception as exc:
                print(f"Item {item} generated an exception: {exc}")
    save_json(new_data_path, new_data)


if __name__ == "__main__":
    main()
