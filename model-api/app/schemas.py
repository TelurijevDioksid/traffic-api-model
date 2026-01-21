from pydantic import BaseModel
from typing import List


class DataInput(BaseModel):
    measurements: List[List[float]]


class PredictionBody(BaseModel):
    data: List[List[float]]


class OnlineTrain(BaseModel):
    """
    data_step: koliko nedavnih mjerenja smatramo novim
    """
    data_step: int = 48
