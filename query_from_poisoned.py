import re
from typing import Dict, List


def parse_synthetic_text(data: str) -> List[Dict[str, str]]:
    def extract_answers(data: str) -> List[re.Match]:
        pattern = re.compile(r"Synthetic Answer:\s*(.+?)(?=\n|$)", re.DOTALL)
        return pattern.findall(data)

    def process_matches(matches: List[re.Match]) -> List[Dict[str, str]]:
        return [{"answer": match.strip()} for match in matches]

    matches = extract_answers(data)
    return process_matches(matches)


# Example usage:
data = """
Synthetic Answer: Paris, France
Synthetic Answer: The city Paris
Synthetic Answer: Paris, French capital
"""
parsed_data = parse_synthetic_text(data)
print(parsed_data)
