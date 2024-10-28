import concurrent.futures
import threading
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel
from tqdm import tqdm
from utils.json_utils import load_json, save_json
from utils.openai_utils import OpenaiQueryHandler


class QA(BaseModel):
    answer: str


class StatementGenerator:
    def __init__(
        self,
        original_data_path: str,
        new_data_path: str,
        max_workers: int = 20,
        max_items: int = 100,
        max_retries: int = 3,
    ):
        """
        Initializes the StatementGenerator with paths and concurrency settings.

        Args:
            original_data_path (str): Path to the original JSON data.
            new_data_path (str): Path to save the new JSON data with statements.
            max_workers (int, optional): Number of threads for parallel processing. Defaults to 20.
            max_items (int, optional): Number of items to process. Defaults to 100.
            max_retries (int, optional): Maximum number of retries for generating statements. Defaults to 3.
        """
        self.original_data_path = original_data_path
        self.new_data_path = new_data_path
        self.max_workers = max_workers
        self.max_items = max_items
        self.max_retries = max_retries
        self.system_statement = self._get_system_statement()

    def _get_system_statement(self) -> str:
        """
        Returns the system statement used as a prompt for the OpenAI model.

        Returns:
            str: The system statement.
        """
        return """
Create five different biography statements about an individual, using the exact provided details. Each statement should be on a new line and include all the given pieces of information without exception.

# Details
- **Name:** Ward Williams
- **Birth Date:** 24 December 1934
- **Birth Place:** Apia, Samoa
- **Death Date:** 06 February 2003

# Steps

1. Formulate each biography statement using all provided pieces of information.
2. Ensure each statement is clear, distinct, and incorporates the exact details without any changes.
3. Each statement should be on a new line.

# Output Format

The output should be five distinct sentences, each on a separate line, maintaining the structure as full sentences.

# Examples

**Input Detail Example:**
- Name: John Doe
- Birth Date: 10 January 1920
- Birth Place: Springfield, USA
- Death Date: 15 March 1980

**Output Example:**
John Doe was born on 10 January 1920 in Springfield, USA and died on 15 March 1980.
Springfield, USA was the birthplace of John Doe, born on 10 January 1920, who passed away on 15 March 1980.
The individual named John Doe, born in Springfield, USA on 10 January 1920, died on 15 March 1980.
On 10 January 1920, John Doe was born in Springfield, USA and he died on 15 March 1980.
John Doe, whose birth was on 10 January 1920 in Springfield, USA, died on 15 March 1980.
"""

    def _get_user_statement(self, name: str, triplet: Dict[str, Any]) -> str:
        """
        Constructs the user statement from the name and triplet data.

        Args:
            name (str): The name of the individual.
            triplet (Dict[str, Any]): A dictionary of key-value pairs with biographical information.

        Returns:
            str: The formatted user statement.
        """
        statement = f"name: {name}"
        for key, value in triplet.items():
            statement += f"\n{key}: {value}"
        return statement

    def _process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single item to generate a biographical statement.

        Args:
            item (Dict[str, Any]): The data item containing biographical information.

        Returns:
            Dict[str, Any]: The processed item with the generated statements as a list.
        """
        name = item["name"]
        triplet = item["triplet"]
        user_statement = self._get_user_statement(name, triplet)

        openai_kwargs = {
            "model": "gpt-4o-mini",
            "max_tokens": 256,
            "top_p": 1,
            "temperature": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "timeout": 10,
        }

        handler = OpenaiQueryHandler(
            system_prompt=self.system_statement,
            user_prompt=user_statement,
            response_format=QA,
            kwargs=openai_kwargs,
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                response = handler.query_with_schema()
                response_text = response.answer.strip()
                statements = [
                    stmt.strip()
                    for stmt in response_text.split("\n")
                    if stmt.strip()
                ]

                if len(statements) == 5:
                    # Successfully got five statements
                    return {
                        "index": item["index"],
                        "name": name,
                        "content": item["content"],
                        "target_text": item["target_text"],
                        "triplet": triplet,
                        "statement": statements,  # Save as list
                        "fake_triplet": item.get("fake_triplet", {}),
                    }
                else:
                    print(
                        f"Attempt {attempt}: Expected 5 statements, but got {len(statements)}. Retrying..."
                    )
            except Exception as exc:
                print(
                    f"Error processing item {item.get('index', 'unknown')} on attempt {attempt}: {exc}"
                )

        # After max retries, save whatever was obtained, even if not five
        print(
            f"Failed to generate exactly 5 statements for item {item.get('index', 'unknown')} after {self.max_retries} attempts."
        )
        return {
            "index": item["index"],
            "name": name,
            "content": item["content"],
            "target_text": item["target_text"],
            "triplet": triplet,
            "statement": (
                statements if "statements" in locals() else []
            ),  # Save whatever was obtained
            "fake_triplet": item.get("fake_triplet", {}),
        }

    def generate_statements(self) -> List[Dict[str, Any]]:
        """
        Generates biographical statements for all items in the original data.

        Returns:
            List[Dict[str, Any]]: A list of processed items with generated statements.
        """
        original_data = load_json(self.original_data_path)[: self.max_items]
        new_data = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_item = {
                executor.submit(self._process_item, item): item
                for item in original_data
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_item),
                total=len(future_to_item),
                desc="Generating Statements",
            ):
                item = future_to_item[future]
                try:
                    result = future.result()
                    new_data.append(result)
                except Exception as exc:
                    print(
                        f"Error processing item {item.get('index', 'unknown')}: {exc}"
                    )

        return new_data

    def save_statements(self, data: List[Dict[str, Any]]) -> None:
        """
        Saves the generated statements to a JSON file.

        Args:
            data (List[Dict[str, Any]]): The list of processed items with statements.
        """
        save_json(self.new_data_path, data)


class CounterfactualStatementGenerator:
    def __init__(
        self,
        statements_data_path: str,
        updated_data_path: str,
        num_counterfactuals: int = 1,
        max_workers: int = 20,
        max_items: int = 100,
    ):
        """
        Initializes the CounterfactualStatementGenerator with paths and concurrency settings.

        Args:
            statements_data_path (str): Path to the JSON data with original statements.
            updated_data_path (str): Path to save the JSON data with counterfactual statements.
            num_counterfactuals (int, optional): Number of counterfactual statements to generate per item. Defaults to 1.
            max_workers (int, optional): Number of threads for parallel processing. Defaults to 20.
            max_items (int, optional): Number of items to process. Defaults to 100.
        """
        self.statements_data_path = statements_data_path
        self.updated_data_path = updated_data_path
        self.num_counterfactuals = num_counterfactuals
        self.max_workers = max_workers
        self.max_items = max_items
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """
        Returns the system prompt used for generating counterfactual statements.

        Returns:
            str: The system prompt.
        """
        return """
Identify the relevant pieces of new information and replace or insert them into the given statement accurately. Maintain the exact wording from the provided details, and ensure each piece of information is clearly identified separately within the statement. Insert all pieces of information, including any additional information provided.

# Steps

1. **Identify Existing Information in the Statement**: Determine which details in the original statement need to be updated.
2. **Identify Relevant New Information**: Review the provided set of information and match it with the components in the statement.
3. **Update the Statement**: Accurately replace or insert the new information into the statement, ensuring each piece is clearly separated.
4. **Use Provided Wording**: Incorporate the exact wording as provided in the related information.
5. **Check Completeness**: Verify all relevant new pieces of information have been included and nothing is overlooked.

# Output Format

- A rewritten sentence with updated and additional information clearly identified.
- Each piece of information should be delineated, ensuring clarity and accuracy.

# Example

## Input
- Statement: Ward Williams was born on 26 June 1923 in Colfax, Indiana, and passed away on 17 December 2005.
- Name: Ward Williams
- Birth Date: 24 December 1934
- Birth Place: Apia, Samoa
- Death Date: 06 February 2003
- Occupation: Farm manager
- Nationality: Samoan
- Additional Information: ""

## Output
Ward Williams, born on 24 December 1934 in Apia, Samoa, and passed away on 06 February 2003, was a Samoan farm manager. 

# Notes

- Maintain the integrity of the original statement's purpose while integrating new details.
- Ensure that dates and places are accurately adjusted to reflect the updated information.
"""

    def _construct_new_info(
        self,
        fake_triplet: Dict[str, List[str]],
        item_index: int,
        cf_index: int,
    ) -> Dict[str, Any]:
        """
        Constructs the new_info dictionary by selecting one fake value per key based on the item and counterfactual indices.

        Args:
            fake_triplet (Dict[str, List[str]]): The fake_triplet dictionary with lists of fake values.
            item_index (int): The index of the current item for round-robin selection.
            cf_index (int): The counterfactual index for current generation.

        Returns:
            Dict[str, Any]: A dictionary with one fake value per key.
        """
        new_info = {}
        for key, values in fake_triplet.items():
            if not values:
                continue
            # Calculate the selection index based on item_index and cf_index
            selection_index = (item_index + cf_index) % len(values)
            selected_value = values[selection_index]
            new_info[key] = selected_value
        return new_info

    def _construct_user_prompt(
        self, statement: str, new_info: Dict[str, Any]
    ) -> str:
        """
        Constructs the user prompt for counterfactual statement generation.

        Args:
            statement (str): The original biographical statement.
            new_info (Dict[str, Any]): A dictionary of new information to update the statement.

        Returns:
            str: The formatted user prompt.
        """
        prompt = f'- Statement: "{statement}"'
        for key, value in new_info.items():
            prompt += f'\n- {key}: "{value}"'
        return prompt

    def _generate_single_counterfactual(
        self, statement: str, new_info: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generates a single counterfactual statement.

        Args:
            statement (str): The original biographical statement.
            new_info (Dict[str, Any]): The new information to update the statement.

        Returns:
            Tuple[str, Dict[str, Any]]: The generated statement and the fake elements used.
        """
        user_prompt = self._construct_user_prompt(statement, new_info)

        openai_kwargs = {
            "model": "gpt-4o-mini",
            "max_tokens": 256,
            "top_p": 1,
            "temperature": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "timeout": 10,
        }

        handler = OpenaiQueryHandler(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_format=QA,
            kwargs=openai_kwargs,
        )

        try:
            response = handler.query_with_schema()
            response_text = response.answer.strip()
            return response_text, new_info
        except Exception as exc:
            print(
                f"Error generating counterfactual for statement: '{statement}'. Error: {exc}"
            )
            return "", new_info

    def _process_item(
        self, item: Dict[str, Any], item_index: int
    ) -> Dict[str, Any]:
        """
        Processes a single item to generate multiple counterfactual statements.

        Args:
            item (Dict[str, Any]): The data item containing the original statement and fake_triplet.
            item_index (int): The index of the current item for fake_triplet selection.

        Returns:
            Dict[str, Any]: The processed item with the generated counterfactual statements.
        """
        original_statement = item.get("statement", "")
        fake_triplet = item.get("fake_triplet", {})

        if not fake_triplet:
            # No fake_triplet to process
            item["counterfactual_statements"] = []
            return item

        # Initialize the counterfactual_statements list
        item["counterfactual_statements"] = []

        for cf_index in range(self.num_counterfactuals):
            # Construct new_info by selecting one fake value per key
            new_info = self._construct_new_info(
                fake_triplet, item_index, cf_index
            )

            if not new_info:
                continue  # Skip if no new information is available

            # Generate a single counterfactual
            statement, fake_elements = self._generate_single_counterfactual(
                original_statement, new_info
            )

            if statement:
                # Append both the statement and the fake elements used
                item["counterfactual_statements"].append(
                    {"statement": statement, "fake_elements": fake_elements}
                )

        return item

    def generate_counterfactuals(self) -> List[Dict[str, Any]]:
        """
        Generates counterfactual statements for all items in the statements data.

        Returns:
            List[Dict[str, Any]]: A list of processed items with counterfactual statements.
        """
        statements_data = load_json(self.statements_data_path)[
            : self.max_items
        ]
        updated_data = []
        lock = threading.Lock()  # To synchronize access to updated_data

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Enumerate to get item index for round-robin selection
            future_to_item = {
                executor.submit(self._process_item, item, idx): (item, idx)
                for idx, item in enumerate(statements_data)
            }

            for future in tqdm(
                concurrent.futures.as_completed(future_to_item),
                total=len(future_to_item),
                desc="Generating Counterfactual Statements",
            ):
                item, idx = future_to_item[future]
                try:
                    result = future.result()
                    with lock:
                        updated_data.append(result)
                except Exception as exc:
                    print(
                        f"Error processing item {item.get('index', 'unknown')}: {exc}"
                    )

        return updated_data

    def save_counterfactuals(self, data: List[Dict[str, Any]]) -> None:
        """
        Saves the updated data with counterfactual statements to a JSON file.

        Args:
            data (List[Dict[str, Any]]): The list of processed items with counterfactual statements.
        """
        for idx, item in enumerate(data):
            item["index"] = idx
        save_json(self.updated_data_path, data)


def main():
    """
    The main function to execute the statement and counterfactual statement generation process.
    """
    # Step 1: Generate original statements
    generator = StatementGenerator(
        original_data_path="wikibio_1027.json",
        new_data_path="wikibio_1027_statement.json",
        max_workers=10,
        max_items=10,
    )
    generated_data = generator.generate_statements()
    generator.save_statements(generated_data)
    print(f"Generated statements saved to {generator.new_data_path}")

    # Step 2: Generate counterfactual statements
    # Each item in 'wikibio_1027_statement.json' should include 'fake_triplet'
    counterfactual_generator = CounterfactualStatementGenerator(
        statements_data_path="wikibio_1027_statement.json",
        updated_data_path="wikibio_1027_statement_with_counterfactual.json",
        num_counterfactuals=5,  # Set desired number of counterfactuals per item here
        max_workers=10,  # Increased workers for higher parallelism
        max_items=10,
    )
    counterfactual_data = counterfactual_generator.generate_counterfactuals()
    counterfactual_generator.save_counterfactuals(counterfactual_data)
    print(
        f"Counterfactual statements saved to {counterfactual_generator.updated_data_path}"
    )


if __name__ == "__main__":
    main()
