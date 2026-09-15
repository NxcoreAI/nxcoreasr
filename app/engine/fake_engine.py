"""测试与无网环境用的假引擎（ASR_FAKE_ENGINE=1）。

固定行为：>=2 秒的音频产出两个说话人各一句；不同音频里 local_spk
相同则音色向量相同，因此跨音频认人在测试里可预期。
"""
import wave

import numpy as np

from .base import Engine, Segment, normalize

EMBED_DIM = 192


def _spk_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return normalize(rng.standard_normal(EMBED_DIM))


class FakeEngine(Engine):
    name = "fake"

    def _duration_ms(self, wav_path: str) -> int:
        with wave.open(wav_path, "rb") as w:
            return int(w.getnframes() / w.getframerate() * 1000)

    def transcribe(self, wav_path: str) -> list[Segment]:
        dur = self._duration_ms(wav_path)
        if dur < 200:
            return []
        if dur < 2000:
            return [Segment(0, dur, "这是一段测试语音。", 0)]
        half = dur // 2
        return [
            Segment(0, half, "你好，我是第一位说话人。", 0),
            Segment(half, dur, "好的，我是第二位说话人。", 1),
        ]

    def embed(self, wav_path: str, spans: list[tuple[int, int]]) -> np.ndarray:
        dur = self._duration_ms(wav_path)
        # 根据片段落在前半还是后半判断它属于哪个假说话人
        mid = dur // 2
        seed = 0 if spans and spans[0][0] < mid else 1
        return _spk_vector(seed)
