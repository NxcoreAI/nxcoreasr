"""FunASR 真引擎：Fun-ASR-Nano（端到端，自带标点与逐字时间戳）+ fsmn-vad + cam++（全 CPU）。

说话人分离不再用 FunASR 内置流程（其 punc_segment 模式依赖外置标点模型，
Fun-ASR-Nano 切过去后没有标点模型，会退化成"整个 VAD 段落一个说话人"），
改为：逐字时间戳 + cam++ 滑动窗口音色聚类 → 按时间线把每个字归到说话人，
在说话人切换点切开文本。
"""
import logging
import wave
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from .base import Engine, Segment, normalize

log = logging.getLogger(__name__)

MAX_EMBED_SECONDS = 20  # 提音色向量最多取 20 秒素材，够用且省时

WIN_MS = 3000           # 音色窗口长：CAM++ 按约 3 秒语音训练，实测窗口越长
                        # 同人相似度越高越紧（1.5s 时同人最低仅 0.49，3s 到 0.72），
                        # 真实远场音频靠它防同人被拆开；代价是边界混音窗变多，
                        # 由字级/段级吸收逻辑兜底
HOP_MS = 1000           # 滑动步进
MIN_TURN_MS = 1000      # 短于 1 秒的说话片段视为抖动，并入邻段
MIN_RUN_MS = 700        # 字级：不足 0.7 秒（一个窗口步进的量级）的同人片段归入时间上
                        # 更近的邻段，吸收说话人切换点附近的窗口边界串扰
SPK_BATCH = 256         # 每批送入 cam++ 的窗口数（超长音频控内存）
SENTENCE_ENDERS = "。？！?!…"

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


def slide_windows(regions: list[tuple[int, int]], win_ms: int = WIN_MS,
                  hop_ms: int = HOP_MS) -> list[tuple[int, int]]:
    """在语音区域内滑窗；尾部不足一窗时向前借齐成整窗，保证音色向量质量。"""
    wins: list[tuple[int, int]] = []
    for a, b in regions:
        if b - a <= win_ms:
            if b > a:
                wins.append((a, b))
            continue
        end = a + win_ms
        while end < b:
            wins.append((end - win_ms, end))
            end += hop_ms
        wins.append((b - win_ms, b))
    return wins


def cluster_windows(embs: np.ndarray, threshold: float) -> list[int]:
    """平均链接层次聚类：簇间平均相似度达标才合并，个别抖动窗口不会独立成簇。
    返回按首次出现顺序编号的簇标签。"""
    n = len(embs)
    if n == 0:
        return []
    if n == 1:
        return [0]
    Z = linkage(embs, method="average", metric="cosine")
    raw = fcluster(Z, t=1.0 - threshold, criterion="distance")
    order: dict[int, int] = {}
    labels = []
    for r in raw:
        if r not in order:
            order[r] = len(order)
        labels.append(order[r])
    return labels


def _merge_turns(windows: list[tuple[int, int]], labels: list[int]) -> list[list]:
    """合并相邻同说话人的窗口为说话片段 [(start, end, spk), ...]。"""
    turns: list[list] = []
    for (a, b), k in zip(windows, labels):
        if turns and turns[-1][2] == k:
            turns[-1][1] = max(turns[-1][1], b)
        else:
            turns.append([a, b, k])
    return turns


def smooth_turns(turns: list[list], min_ms: int = MIN_TURN_MS) -> list[list]:
    """过短的说话片段并入更长的一侧邻段，再合并相邻同人片段。"""
    turns = [list(t) for t in turns]
    changed = True
    while changed and len(turns) > 1:
        changed = False
        for i, t in enumerate(turns):
            if t[1] - t[0] >= min_ms:
                continue
            prev_t = turns[i - 1] if i > 0 else None
            next_t = turns[i + 1] if i + 1 < len(turns) else None
            if prev_t is None:
                target = next_t
            elif next_t is None:
                target = prev_t
            else:
                target = prev_t if prev_t[1] - prev_t[0] >= next_t[1] - next_t[0] else next_t
            target[0] = min(target[0], t[0])
            target[1] = max(target[1], t[1])
            turns.pop(i)
            changed = True
            break
    merged: list[list] = []
    for a, b, k in turns:
        if merged and merged[-1][2] == k:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b, k])
    return merged


PAUSE_MARKS = "，、；,;"  # 次级停顿：说话人切换也常发生在这些短语边界


def resolve_turn_overlaps(turns: list[list], tokens: list[dict],
                          snap_ms: int = HOP_MS) -> list[list]:
    """相邻异人轮次因滑窗重叠会时间交叠（3s 窗 1s 步进最多叠 2s），字级
    "最大重叠"裁决会把切换点附近的字系统性划给先发言人。

    处理：把交叠段在句读边界处切开——优先句末标点、其次停顿标点（候选可
    比后一轮起点再向前看 snap_ms，混叠窗把切换点裹进前一轮），取离交叠
    中点最近的边界；无标点可用才退回几何中点。说话人切换几乎总发生在
    短语边界，比窗口几何中点准（实测"我看着你的脸，|是嘉应学院"的切换
    点被拖影 1.4s，句读吸附后归位）。
    """
    if len(turns) < 2:
        return turns
    out = [list(t) for t in turns]
    for i in range(len(out) - 1):
        a1, b1, k1 = out[i]
        a2, b2, k2 = out[i + 1]
        if k1 == k2 or b1 <= a2:
            continue
        mid = (a2 + b1) / 2
        best_cut, best_d = None, None
        for tok in tokens:
            e = tok["end"]
            if a2 - snap_ms <= e <= b1 and tok["token"] in SENTENCE_ENDERS + PAUSE_MARKS:
                d = abs(e - mid)
                # 句末标点优先（距离乘小系数），同距取句末
                d *= 0.5 if tok["token"] in SENTENCE_ENDERS else 1.0
                if best_d is None or d < best_d:
                    best_cut, best_d = e, d
        cut = int(best_cut) if best_cut is not None else int(mid)
        out[i][1] = cut
        out[i + 1][0] = cut
    return out


def token_speakers(tokens: list[dict], turns: list[list]) -> list[int]:
    """每个字按时间重叠最多的说话片段归人；标点没有独立音色，跟随邻侧的字。"""
    out: list[int | None] = [None] * len(tokens)
    for i, tok in enumerate(tokens):
        if not tok["token"].isalnum():
            continue
        a, b = tok["start"], tok["end"]
        best_k, best_ov = None, 0
        for ts, te, k in turns:
            ov = min(b, te) - max(a, ts)
            if ov > best_ov:
                best_ov, best_k = ov, k
        if best_k is None:
            best_k = out[i - 1] if i > 0 and out[i - 1] is not None else 0
        out[i] = best_k
    for i in range(len(out)):  # 标点跟随前一个字；开头标点跟随后一个字
        if out[i] is None:
            out[i] = out[i - 1] if i > 0 else (out[i + 1] if i + 1 < len(out) else 0)
    return out  # type: ignore[return-value]


def smooth_token_runs(tokens: list[dict], speakers: list[int],
                      min_ms: int = MIN_RUN_MS) -> list[int]:
    """同人连续字构成 run；短于 min_ms 的 run 并入时间上更近的一侧，消除边界抖动。
    串扰都发生在说话人切换点附近，所以按间隙距离而不是片段长短选归属侧。"""
    runs: list[list] = []  # [start_idx, end_idx, spk]
    for i, k in enumerate(speakers):
        if runs and runs[-1][2] == k:
            runs[-1][1] = i
        else:
            runs.append([i, i, k])
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for r_i, r in enumerate(runs):
            a = tokens[r[0]]["start"]
            b = tokens[r[1]]["end"]
            if b - a >= min_ms:
                continue
            prev_r = runs[r_i - 1] if r_i > 0 else None
            next_r = runs[r_i + 1] if r_i + 1 < len(runs) else None
            if prev_r is None:
                target = next_r
            elif next_r is None:
                target = prev_r
            else:
                gap_prev = a - tokens[prev_r[1]]["end"]
                gap_next = tokens[next_r[0]]["start"] - b
                target = prev_r if gap_prev < gap_next else next_r
            target[0] = min(target[0], r[0])
            target[1] = max(target[1], r[1])
            runs.pop(r_i)
            changed = True
            break
    merged: list[list] = []
    for r in runs:
        if merged and merged[-1][2] == r[2]:
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(r)
    out = list(speakers)
    for s, e, k in merged:
        for i in range(s, e + 1):
            out[i] = k
    return out


def group_segments(tokens: list[dict], speakers: list[int]) -> list[Segment]:
    """连续同说话人的字合成一段；句末标点后再切，保持句子粒度。"""
    segs: list[Segment] = []
    cur: list[dict] = []
    cur_spk = -1

    def flush():
        if not cur:
            return
        segs.append(Segment(
            start_ms=cur[0]["start"],
            end_ms=cur[-1]["end"],
            text="".join(t["token"] for t in cur),
            local_spk=cur_spk,
        ))

    for tok, k in zip(tokens, speakers):
        if cur and (k != cur_spk or cur[-1]["token"][-1] in SENTENCE_ENDERS):
            flush()
            cur = []
        cur.append(tok)
        cur_spk = k
    flush()
    return segs


def merge_short_fragments(segs: list[Segment], min_ms: int = MIN_RUN_MS) -> list[Segment]:
    """不足 min_ms 的碎片段（说话人切换点附近的边角字）按时间就近并入邻段。"""
    out = [s for s in segs]
    changed = True
    while changed and len(out) > 1:
        changed = False
        for i, s in enumerate(out):
            if s.end_ms - s.start_ms >= min_ms:
                continue
            prev_s = out[i - 1] if i > 0 else None
            next_s = out[i + 1] if i + 1 < len(out) else None
            if prev_s is None:
                target, ti = next_s, i + 1
            elif next_s is None:
                target, ti = prev_s, i - 1
            else:
                gap_prev = s.start_ms - prev_s.end_ms
                gap_next = next_s.start_ms - s.end_ms
                if gap_prev < gap_next:
                    target, ti = prev_s, i - 1
                else:
                    target, ti = next_s, i + 1
            if ti < i:
                target.text += s.text
                target.end_ms = max(target.end_ms, s.end_ms)
            else:
                target.text = s.text + target.text
                target.start_ms = min(target.start_ms, s.start_ms)
            out.pop(i)
            changed = True
            break
    return out


class FunasrEngine(Engine):
    name = "funasr"

    def __init__(self, model_dir: Path, window_threshold: float = 0.65,
                 llm_quantize: bool = False):
        from funasr import AutoModel  # 延迟导入，fake 模式不需要装

        log.info("加载 FunASR 模型（本地 %s）...", model_dir)
        self.window_threshold = window_threshold  # 窗口簇间平均相似度达到该值才合并：
        # 单窗口抖动大，用平均链接聚类的簇间相似度判断，比单窗比对稳。
        # 偏高拆人、偏低并人，拿真实音频跑 app/diarize_report.py 校准
        self.asr = AutoModel(
            model=_local(model_dir, "fun-asr-nano"),
            vad_model=_local(model_dir, "fsmn-vad"),
            vad_kwargs={"max_single_segment_time": 30000},
            device="cpu",
            disable_update=True,
        )
        if llm_quantize:
            # 只量化 LLM 分支（文本生成，参数量大头）。逐字时间戳由 CTC 分支
            # 独立产出、cam++/vad 是独立模型，说话人分离不受影响
            import platform

            import torch
            engines = [e for e in torch.backends.quantized.supported_engines
                       if e != "noqengine"]
            if not engines:  # 极少数无量化后端的 torch 构建，放弃量化保启动
                log.warning("当前 torch 构建无量化后端，跳过 LLM 量化")
            else:
                # x86 用 onednn/fbgemm（qnnpack 在 x86 上是通用实现，实测慢
                # 一个量级）；arm 用 qnnpack（NEON/AMX 原生路径）
                prefer = ("onednn", "fbgemm", "qnnpack") \
                    if platform.machine() in ("x86_64", "AMD64") \
                    else ("qnnpack", "onednn", "fbgemm")
                engine = next((e for e in prefer if e in engines), engines[0])
                torch.backends.quantized.engine = engine

                def _rss_mb():  # 量化各阶段的内存取证（Linux 读 /proc）
                    try:
                        with open("/proc/self/status") as f:
                            for line in f:
                                if line.startswith("VmRSS:"):
                                    return int(line.split()[1]) / 1024
                    except OSError:
                        pass
                    return -1

                log.info("量化前 RSS %.0fMB", _rss_mb())
                llm = self.asr.model.llm
                self.asr.model.llm = torch.ao.quantization.quantize_dynamic(
                    llm, {torch.nn.Linear}, dtype=torch.qint8)
                del llm  # 释放原 bf16 权重
                import gc
                gc.collect()
                log.info("LLM 分支已 int8 动态量化（引擎 %s），量化后 RSS %.0fMB",
                         engine, _rss_mb())
        self.vad = AutoModel(model=_local(model_dir, "fsmn-vad"), device="cpu",
                             disable_update=True)
        self.spk = AutoModel(model=_local(model_dir, "cam-plus"), device="cpu",
                             disable_update=True)
        log.info("FunASR 模型加载完成")

    def _transcribe_raw(self, wav_path: str) -> tuple[list[dict], str]:
        res = self.asr.generate(input=wav_path, batch_size_s=300,
                                output_timestamp=True, return_time_stamps=True)
        if not res:
            return [], ""
        raw_tokens = res[0].get("timestamps") or []
        tokens = [{"token": t["token"], "start": int(t["start_time"] * 1000),
                   "end": int(t["end_time"] * 1000)} for t in raw_tokens]
        return tokens, (res[0].get("text") or "").strip()

    def raw_tokens(self, wav_path: str) -> list[dict] | None:
        """转写并返回逐字时间戳 [{"token","start","end"(ms)}, ...]。

        无时间戳输出时返回 None（调用方退化为整段单说话人）。
        与说话人阈值无关，diarize_eval 缓存复用。
        """
        tokens, _ = self._transcribe_raw(wav_path)
        return tokens or None

    def transcribe(self, wav_path: str) -> list[Segment]:
        tokens, text = self._transcribe_raw(wav_path)
        if not tokens:
            # 没有逐字时间戳时退化为整段单说话人
            if not text:
                return []
            with wave.open(wav_path, "rb") as w:
                dur = int(w.getnframes() / w.getframerate() * 1000)
            return [Segment(0, dur, text, 0)]
        return self._diarize(wav_path, tokens)

    def _diarize(self, wav_path: str, tokens: list[dict]) -> list[Segment]:
        """逐字时间戳 + cam++ 滑窗聚类 -> 按时间线归人、切分说话片段。"""
        vr = self.vad.generate(input=wav_path)[0]
        regions = [(int(a), int(b)) for a, b in vr.get("value") or []]
        wins = slide_windows(regions)
        labels = []
        if wins:
            pcm, rate = self._load_pcm(wav_path)
            slices = [pcm[a * rate // 1000: b * rate // 1000] for a, b in wins]
            embs = []
            for j in range(0, len(slices), SPK_BATCH):  # 超长音频分批提音色，控内存
                for r in self.spk.generate(input=slices[j:j + SPK_BATCH], fs=rate):
                    emb = r["spk_embedding"]
                    if hasattr(emb, "detach"):  # torch tensor
                        emb = emb.detach().cpu().numpy()
                    embs.append(normalize(np.asarray(emb).reshape(-1)))
            labels = cluster_windows(np.stack(embs), self.window_threshold)
        turns = smooth_turns(_merge_turns(wins, labels)) if wins else []
        turns = resolve_turn_overlaps(turns, tokens) if turns else []
        speakers = smooth_token_runs(tokens, token_speakers(tokens, turns))
        return merge_short_fragments(group_segments(tokens, speakers))

    def _load_pcm(self, wav_path: str) -> tuple[np.ndarray, int]:
        with wave.open(wav_path, "rb") as w:
            rate = w.getframerate()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return data.astype(np.float32) / 32768.0, rate

    def embed(self, wav_path: str, spans: list[tuple[int, int]]) -> np.ndarray:
        pcm, rate = self._load_pcm(wav_path)
        pieces = []
        total = 0
        # 按 span 时长降序取素材：长段是持续说话的干净音频，靠前的短段
        # 常是切换点附近的混音碎片，会污染认人向量
        for start_ms, end_ms in sorted(spans, key=lambda s: s[1] - s[0], reverse=True):
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
