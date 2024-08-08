import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import time
from typing import Annotated, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, conlist


@dataclass
class MyData(BaseModel):
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

    def query_with_schema(self) -> None:
        max_attempts = self.kwargs.get("max_attempts", 30)
        attempts = 1
        while attempts <= max_attempts:
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=self.kwargs.get("model", "gpt-4o-mini"),
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
                print(f"{e=}")
                print(f"Attempt {attempts} failed.")
                attempts += 1
                if attempts > max_attempts:
                    raise e


if __name__ == "__main__":
    system_prompt = "Generate paraphrased contexts. Maintain the length of the context. Each context should be paraphrased versions of the following context:"
    user_prompt = """context:\n was written by Reese and Wernick and played in front of \"Logan\". \"Deadpool 2\" was released on May 18, 2018, with Baccarin, T. J. Miller, Uggams, Hildebrand, and Kapičić all returning. Josh Brolin joined them as Cable. The film explores the team X-Force, which includes Deadpool and Cable. In March 2017, Reese said that a future film focused on that group would be separate from \"Deadpool 3\", \"so I think we'll be able to take two paths. [\"X-Force\"] is where we're launching something bigger, but then [\"Deadpool 3\" is] where we're contracting and staying personal and small.\" After the acquisition\n Provide the 5 contexts."""
    query_handler = OpenaiQueryHandler(system_prompt, user_prompt)
    response = query_handler.query_with_schema()
    print(response)
    print(response.contexts)
    print(len(response.contexts))
