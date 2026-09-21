"""llama.cpp 混合引擎：重活交给 C++，Python 只做轻量收尾。

- llama-funasr-cli（GGUF 量化版）负责 VAD 分段 + 音频编码器 + LLM 解码出文本，
  并通过 --dump-enc 把编码器中间输出落盘（需带补丁的构建，官方二进制没有）
- Python 侧只保留 funasr 的 CTC 分支（一个小投影层），对落盘的编码器输出做
  强制对齐，补回逐字时间戳——GGUF 没带 CTC，但 CTC 只吃编码器输出，与文本
  由谁生成无关
- 说话人分离复用 funasr_engine 的 cam++ 滑窗聚类管线
"""
import gc
import json
import logging
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from .base import Segment
from .funasr_engine import FunasrEngine, _local

log = logging.getLogger(__name__)

CTC_FRAME_MS = 60  # CTC 帧格：LFR_N=6 × 10ms

GGUF_ENC = "funasr-encoder-f16.gguf"
GGUF_VAD = "fsmn-vad.gguf"


class LlamaCppEngine(FunasrEngine):
    name = "llamacpp"

    def __init__(self, model_dir: Path, bin_path: str, gguf_dir: str,
                 llm_gguf: str = "qwen3-0.6b-q4km.gguf",
                 window_threshold: float = 0.65):
        from funasr import AutoModel

        log.info("加载 llama.cpp 引擎（CLI %s，GGUF %s）...", bin_path, gguf_dir)
        self.bin_path = str(bin_path)
        self.enc_gguf = str(Path(gguf_dir) / GGUF_ENC)
        self.llm_gguf = str(Path(gguf_dir) / llm_gguf)
        self.vad_gguf = str(Path(gguf_dir) / GGUF_VAD)
        self.window_threshold = window_threshold

        # 只为 CTC 对齐与说话人分离加载 funasr；LLM/编码器在 llama.cpp 侧，
        # 加载后立刻卸掉，省约 3G 内存
        self.asr = AutoModel(model=_local(model_dir, "fun-asr-nano"),
                             device="cpu", disable_update=True)
        inner = self.asr.model
        del inner.llm, inner.audio_encoder, inner.audio_adaptor
        gc.collect()
        self.ctc_decoder = inner.ctc_decoder
        self.ctc = inner.ctc
        self.ctc_tokenizer = inner.ctc_tokenizer
        self.blank_id = inner.blank_id

        self.vad = AutoModel(model=_local(model_dir, "fsmn-vad"), device="cpu",
                             disable_update=True)
        self.spk = AutoModel(model=_local(model_dir, "cam-plus"), device="cpu",
                             disable_update=True)
        log.info("llama.cpp 引擎加载完成")

    def _cli_run(self, wav_path: str) -> tuple[list[dict], list[dict] | None]:
        """跑 llama-funasr-cli：返回 (VAD 段落, 逐字时间戳或 None)。

        对齐必须在临时目录内完成——编码器落盘文件与目录同生命周期。
        二进制没带落盘补丁时拿不到时间戳，调用方退化为整段单说话人。
        """
        with tempfile.TemporaryDirectory(prefix="llamacpp_asr_") as tmp:
            dump_prefix = str(Path(tmp) / "seg")
            cmd = [self.bin_path, "--enc", self.enc_gguf, "-m", self.llm_gguf,
                   "-a", wav_path, "--vad", self.vad_gguf,
                   "--dump-enc", dump_prefix]
            log.info("llama.cpp 转写: %s", wav_path)
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=7200)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"llama-funasr-cli 失败({proc.returncode}): {proc.stderr[-2000:]}")
            segs = [json.loads(line) for line in proc.stdout.splitlines()
                    if line.startswith("{")]
            segs = [s for s in segs if s.get("text", "").strip()]
            if not segs:
                return [], None
            if not Path(f"{dump_prefix}.0.enc").exists():
                log.warning("CLI 未产出编码器落盘，退化为整段单说话人"
                            "（换带补丁的构建可补回时间戳）")
                return segs, None
            # 落盘文件与临时目录同生命周期，对齐须在目录删除前完成
            tokens: list[dict] = []
            for i, s in enumerate(segs):
                tokens += self._align_segment(f"{dump_prefix}.{i}.enc",
                                              s["text"].strip(), s["start_ms"])
            return segs, tokens or None

    def raw_tokens(self, wav_path: str) -> list[dict] | None:
        """转写并返回逐字时间戳 [{"token","start","end"(ms)}, ...]。

        无法对齐时返回 None。与说话人阈值无关，diarize_eval 缓存复用。
        """
        return self._cli_run(wav_path)[1]

    def transcribe(self, wav_path: str) -> list[Segment]:
        segs, tokens = self._cli_run(wav_path)
        if not segs:
            return []
        if not tokens:
            return [Segment(s["start_ms"], s["end_ms"], s["text"].strip(), 0)
                    for s in segs]
        return self._diarize(wav_path, tokens)

    def _align_segment(self, dump_path: str, text: str,
                       offset_ms: int) -> list[dict]:
        """读编码器落盘 -> CTC 前向 -> 强制对齐出逐字时间戳。"""
        from funasr.models.fun_asr_nano.tools.utils import forced_align

        with open(dump_path, "rb") as f:
            t_len, d_model = struct.unpack("<ii", f.read(8))
            arr = np.frombuffer(f.read(), dtype="<f4").reshape(t_len, d_model)
        enc = torch.from_numpy(arr.copy()).unsqueeze(0)
        dec_out, _ = self.ctc_decoder(enc, torch.tensor([t_len]))
        log_probs = self.ctc.log_softmax(dec_out)[0]
        target_ids = torch.tensor(self.ctc_tokenizer.encode(text),
                                  dtype=torch.int64)
        out = []
        for t in forced_align(log_probs, target_ids, self.blank_id):
            out.append({
                "token": self.ctc_tokenizer.decode([t["token"]]),
                "start": offset_ms + int(t["start_time"]) * CTC_FRAME_MS,
                "end": offset_ms + int(t["end_time"]) * CTC_FRAME_MS,
            })
        return out
