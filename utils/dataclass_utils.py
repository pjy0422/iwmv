from dataclasses import dataclass
from typing import List

from pydantic import BaseModel


@dataclass
class CF_Answers(BaseModel):
    answers: List[str]


@dataclass
class CF_Contexts(BaseModel):
    contexts: List[str]


@dataclass
class CF_Cleaning(BaseModel):
    steps: str
    texts: List[str]


@dataclass
class Paraphrase(BaseModel):
    contexts: List[str]


@dataclass
class QA(BaseModel):
    answer: str
