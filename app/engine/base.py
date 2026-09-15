"""转写引擎接口。输入统一是 16k 单声道 PCM16 wav。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    text: str
    local_spk: int  # 本段音频内的说话人编号，跨音频无意义


class Engine(ABC):
    name: str = "base"

    @abstractmethod
    def transcribe(self, wav_path: str) -> list[Segment]:
        """整段转写 + 音频内说话人分离。"""

    @abstractmethod
    def embed(self, wav_path: str, spans: list[tuple[int, int]]) -> np.ndarray:
        """对给定时间片段（毫秒）提取归一化音色向量。"""


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v
