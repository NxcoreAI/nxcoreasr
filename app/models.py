"""请求体模型。"""
from pydantic import BaseModel, Field


class RegisterSessionReq(BaseModel):
    external_id: str = Field(min_length=1, max_length=128)


class CreateJobReq(BaseModel):
    external_id: str = Field(min_length=1, max_length=128)
    # 用户专属热词（人物库姓名/别名/组织，≤200 词）：转写落库前做拼音同音
    # 纠偏（引擎无关，funasr/llamacpp 统一生效，见 app/hotwords.py）。
    hotwords: list[str] | None = Field(default=None, max_length=200)


class NameReq(BaseModel):
    name: str | None = None  # None 或空串 = 清除人工命名


class MergeReq(BaseModel):
    from_person: str
    into_person: str
