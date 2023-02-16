from pydantic import BaseModel, Field
from typing import Dict



class BiologicalSexSettings(BaseModel):
    context: str = Field(..., alias="CONTEXT")
    scenarios: str = Field(..., alias="SCENARIOS")
    score_colname: str = Field(..., alias="SCORE_COLNAME")
    mapping: Dict[str, str] = Field(..., alias="MAPPING")
