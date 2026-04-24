from enum import Enum
from typing import Literal
from pydantic import BaseModel


class ReviewItemType(str, Enum):
    vocabulary_flag = "vocabulary_flag"
    passage_candidate = "passage_candidate"
    term_candidate = "term_candidate"


class ReviewItem(BaseModel):
    id: str
    type: ReviewItemType
    label: str
    context: str
    payload: dict


class Decision(BaseModel):
    item_id: str
    action: Literal["accept", "reject"]
