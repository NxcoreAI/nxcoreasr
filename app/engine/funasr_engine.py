"""FunASR 真引擎：Fun-ASR-Nano（端到端，自带标点）+ fsmn-vad + cam++（全 CPU）。

模型从本地 model_dir 加载；目录缺失时启动阶段自动从 modelscope
下载到该目录（首次约 4GB，之后不再联网）。
"""
import logging
import wave
from pathlib import Path

import numpy as np

from .base import Engine, Segment, normalize

log = logging.getLogger(__name__)

MAX_EMBED_SECONDS = 20  # 提音色向量最多取 20 秒素材，够用且省时

# 模型目录名 -> modelscope 上的下载标识
MODELS = {
    "fun-asr-nano": "FunAudioLLM/Fun-ASR-Nano-2512",
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "cam-plus": "iic/speech_campplus_sv_zh-cn_16k-common",
}


def _local(model_dir: Path, name: str) -> str:
    p = model_dir / name
    if not (p / "configuration.json").exists():
        log.info("本地模型缺失，自动下载 %s -> %s（仅首次）", MODELS[name], p)
        from modelscope.hub.snapshot_download import snapshot_download
        snapshot_download(MODELS[name], local_dir=str(p))
    return str(p)


class FunasrEngine(Engine):
    name = "funasr"

    def __init__(self, model_dir: Path):
        from funasr import AutoModel  # 延迟导入，fake 模式不需要装

        log.info("加载 FunASR 模型（本地 %s）...", model_dir)
        self.asr = AutoModel(
            model=_local(model_dir, "fun-asr-nano"),
            vad_model=_local(model_dir, "fsmn-vad"),
            vad_kwargs={"max_single_segment_time": 30000},
            spk_model=_local(model_dir, "cam-plus"),
            device="cpu",
            disable_update=True,
        )
        self.spk = AutoModel(model=_local(model_dir, "cam-plus"), device="cpu",
                             disable_update=True)
        log.info("FunASR 模型加载完成")

    def transcribe(self, wav_path: str) -> list[Segment]:
        res = self.asr.generate(input=wav_path, batch_size_s=300)
        if not res:
            return []
        first = res[0]
        info = first.get("sentence_info") or []
        segments = []
        for s in info:
            text = (s.get("text") or "").strip()
            if not text:
                continue
            segments.append(Segment(
                start_ms=int(s.get("start", 0)),
                end_ms=int(s.get("end", 0)),
                text=text,
                local_spk=int(s.get("spk", 0)),
            ))
        if not segments:
            # 没有句级信息时退化为整段单说话人
            text = (first.get("text") or "").strip()
            if text:
                with wave.open(wav_path, "rb") as w:
                    dur = int(w.getnframes() / w.getframerate() * 1000)
                segments.append(Segment(0, dur, text, 0))
        return segments

    def _load_pcm(self, wav_path: str) -> tuple[np.ndarray, int]:
        with wave.open(wav_path, "rb") as w:
            rate = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return data.astype(np.float32) / 32768.0, rate

    def embed(self, wav_path: str, spans: list[tuple[int, int]]) -> np.ndarray:
        pcm, rate = self._load_pcm(wav_path)
        pieces = []
        total = 0
        for start_ms, end_ms in spans:
            a, b = int(start_ms * rate / 1000), int(end_ms * rate / 1000)
            piece = pcm[a:b]
            pieces.append(piece)
            total += len(piece)
            if total >= MAX_EMBED_SECONDS * rate:
                break
        audio = np.concatenate(pieces) if pieces else pcm
        res = self.spk.generate(input=audio, fs=rate)
        emb = res[0]["spk_embedding"]
        if hasattr(emb, "detach"):  # torch tensor
            emb = emb.detach().cpu().numpy()
        return normalize(np.asarray(emb).reshape(-1))
