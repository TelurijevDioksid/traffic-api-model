from pydantic import BaseModel
from typing import List


class DataInput(BaseModel):
    measurements: List[List[float]]


class PredictionBody(BaseModel):
    data: List[List[float]]
