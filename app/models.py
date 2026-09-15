"""请求体模型。"""
from pydantic import BaseModel, Field


class RegisterSessionReq(BaseModel):
    external_id: str = Field(min_length=1, max_length=128)


class CreateJobReq(BaseModel):
    external_id: str = Field(min_length=1, max_length=128)


class NameReq(BaseModel):
    name: str | None = None  # None 或空串 = 清除人工命名


class MergeReq(BaseModel):
    from_person: str
    into_person: str
