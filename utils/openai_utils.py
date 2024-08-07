import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, conlist


class MyData(BaseModel):
    answers: List[str]
    contexts: List[str]


class OpenaiQueryHandler:
    def __init__(
        self, system_prompt: str, user_prompt: str, **kwargs: Dict
    ) -> None:
        load_dotenv(verbose=True)
        _api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=_api_key)
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.kwargs = kwargs
        self.max_attempts = self.kwargs.get("max_attempts", 30)

    def query_with_schema(self) -> Dict:
        attempts = 1
        while attempts <= self.max_attempts:
            try:
                completion = self.client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.user_prompt},
                    ],
                    max_tokens=self.kwargs.get("max_tokens", 4000),
                    top_p=self.kwargs.get("top_p", 1),
                    temperature=self.kwargs.get("temperature", 0.8),
                    frequency_penalty=self.kwargs.get("frequency_penalty", 0),
                    presence_penalty=self.kwargs.get("presence_penalty", 0),
                    response_format=self.kwargs.get("response_format", MyData),
                )
                return completion.choices[0].message.parsed
            except Exception as e:
                attempts += 1
                if attempts > self.max_attempts:
                    raise e


if __name__ == "__main__":
    system_prompt = "You are given a question and asked to provide 5 answers and relevant 5 contexts."
    user_prompt = "What is human life expectancy in the United States?"
    query_handler = OpenaiQueryHandler(system_prompt, user_prompt)
    print(query_handler.query_with_schema())
