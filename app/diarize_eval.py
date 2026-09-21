"""说话人分离 data case 评测与调参：重放线上管线、对 truth 打分、阈值网格扫描。

用法（case 目录布局见 truth-draft 输出的注释）：
  python -m app.diarize_eval <case_dir> prepare [--config config.yaml]
      转码分片到 cache/、按解码时长重建 full.wav（分片→全时间轴偏移精确对齐）、
      缓存每片与全量的逐字 tokens + 窗口音色向量（一次较慢，之后全走缓存）
  python -m app.diarize_eval <case_dir> replay [--window-t 0.65] [--spk-t 0.45]
      [--min-enroll-ms 6000] [--gray-margin 0.10] [--name run1]
      按线上路径逐片重放（窗口聚类→音频内并簇→跨片三段式认人），决策与结果
      写 runs/<name>/；有 truth.jsonl 时输出 DER 与混淆
  python -m app.diarize_eval <case_dir> sweep --window-t 0.4,0.5,0.65
      [--spk-t 0.45,0.55,...] [--min-enroll-ms 6000,3000]
      阈值网格，全走缓存；按人数命中+DER 排序写 runs/sweep.md
  python -m app.diarize_eval <case_dir> truth-draft [--window-t 0.40]
      用全量音频的窗口聚类生成草稿 truth.draft.jsonl（说话人 A/B/C 按素材量
      命名）；人工校对 speaker 字段后另存 truth.jsonl 生效
"""
import argparse
import json
import subprocess
import wave
from pathlib import Path

import numpy as np


def _iter_chunks(case: Path):
    return sorted((case / "chunks").glob("[0-9][0-9].m4a"))


def _wav_ms(wav: Path) -> int:
    with wave.open(str(wav), "rb") as w:
        return int(w.getnframes() / w.getframerate() * 1000)


def _load_cache(case: Path, name: str):
    npz = np.load(case / "cache" / f"{name}.npz", allow_pickle=False)
    tokens = json.loads(str(npz["tokens"]))
    return (tokens, npz["embs"], npz["starts"].tolist(), npz["ends"].tolist(),
            int(npz["dur_ms"]))


def _segments_from_cache(tokens, embs, starts, ends, window_t: float):
    """缓存重放引擎侧：窗口聚类 -> turns -> 字归人 -> Segment 列表。"""
    from .engine.funasr_engine import (_merge_turns, cluster_windows,
                                       group_segments, merge_short_fragments,
                                       smooth_token_runs, smooth_turns,
                                       token_speakers)
    wins = list(zip(starts, ends))
    labels = cluster_windows(embs, window_t) if len(wins) else []
    turns = smooth_turns(_merge_turns(wins, labels)) if wins else []
    speakers = smooth_token_runs(tokens, token_speakers(tokens, turns))
    return merge_short_fragments(group_segments(tokens, speakers))


def _window_embeddings(engine, wav: str):
    """VAD 滑窗提音色向量（与 diarize_report 同一套，空语音返回空矩阵）。"""
    from .engine.base import normalize
    from .engine.funasr_engine import SPK_BATCH, slide_windows
    vr = engine.vad.generate(input=wav)[0]
    regions = [(int(a), int(b)) for a, b in vr.get("value") or []]
    wins = slide_windows(regions)
    if not wins:
        return [], np.zeros((0, 1))
    pcm, rate = engine._load_pcm(wav)
    slices = [pcm[a * rate // 1000: b * rate // 1000] for a, b in wins]
    embs = []
    for j in range(0, len(slices), SPK_BATCH):
        for r in engine.spk.generate(input=slices[j:j + SPK_BATCH], fs=rate):
            e = r["spk_embedding"]
            if hasattr(e, "detach"):
                e = e.detach().cpu().numpy()
            embs.append(normalize(np.asarray(e).reshape(-1)))
    return wins, np.stack(embs)


class _SpkEmbedCache:
    """按 (片, 窗口阈值) 缓存 local_spk -> 聚合音色向量，扫描不重跑 cam++。

    键是 spans 集合的哈希：local_spk 编号在换阈值后会变，spans 才稳定。
    """

    def __init__(self, case: Path, engine, window_t: float):
        self.case, self.engine, self.tag = case, engine, f"wt{window_t:.2f}"

    def lookup(self, name: str, spans_by_spk: dict[int, list]):
        path = self.case / "cache" / f"{name}.spkemb.{self.tag}.npz"
        by_key: dict[int, tuple[int, np.ndarray, int]] = {}
        if path.exists():
            old = np.load(path, allow_pickle=False)
            by_key = {int(k): (int(old[f"{k}_h"]), old[k],
                               int(old[f"{k}_ms"])) for k in
                      {f.split("_")[0] for f in old.files if "_" not in f}}
        changed = False
        out: dict[int, tuple[np.ndarray, int]] = {}
        for spk, spans in spans_by_spk.items():
            h = hash(tuple(sorted(spans)))
            if spk not in by_key or by_key[spk][0] != h:
                emb = self.engine.embed(
                    str(self.case / "cache" / f"{name}.wav"), spans)
                by_key[spk] = (h, emb, sum(e - b for b, e in spans))
                changed = True
            _, emb, ms = by_key[spk]
            out[spk] = (emb, ms)
        if changed:
            np.savez(path, **{f"{k}": v[1] for k, v in by_key.items()},
                     **{f"{k}_h": v[0] for k, v in by_key.items()},
                     **{f"{k}_ms": v[2] for k, v in by_key.items()})
        return out


def cmd_prepare(args):
    case = Path(args.case)
    (case / "cache").mkdir(exist_ok=True)
    (case / "runs").mkdir(exist_ok=True)
    from .config import load_config
    from .main import _build_engine_
    engine = _build_engine_(load_config(args.config))

    chunks = _iter_chunks(case)
    if not chunks:
        raise SystemExit(f"未找到分片：{case}/chunks/NN.m4a")
    offsets, durs, frames, rate = [], [], [], None
    for i, src in enumerate(chunks, 1):
        wav = case / "cache" / f"{i:02d}.wav"
        if not wav.exists():
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                            "-i", src, "-ac", "1", "-ar", "16000",
                            "-sample_fmt", "s16", str(wav)], check=True)
        with wave.open(str(wav), "rb") as w:
            r = w.getframerate()
            frames.append(w.readframes(w.getnframes()))
        rate = rate or r
        assert r == rate, "分片采样率不一致"
        durs.append(_wav_ms(wav))
        offsets.append(sum(durs[:-1]))
        _prepare_one(engine, case, f"{i:02d}", wav)
        print(f"  分片 {i:02d}/{len(chunks)} 就绪（{durs[-1] / 1000:.1f}s）")

    full = case / "cache" / "full.wav"
    with wave.open(str(full), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(frames))
    _prepare_one(engine, case, "full", full)
    (case / "cache" / "offsets.json").write_text(json.dumps(offsets))
    print(f"full.wav 重建完成（{sum(durs) / 1000:.1f}s），"
          f"分片偏移写 cache/offsets.json")


def _prepare_one(engine, case: Path, name: str, wav: Path):
    out = case / "cache" / f"{name}.npz"
    if out.exists():
        return
    tokens = engine.raw_tokens(str(wav)) or []
    wins, embs = _window_embeddings(engine, str(wav))
    np.savez(out, tokens=json.dumps(tokens, ensure_ascii=False),
             embs=embs, starts=np.array([a for a, _ in wins]),
             ends=np.array([b for _, b in wins]),
             dur_ms=_wav_ms(wav))


def replay_case(case: Path, engine, window_t: float, spk_t: float,
                min_enroll_ms: float, gray_margin: float, name: str):
    """逐片重放线上管线，返回 (决策日志, 全时间轴 utterances, 最终人数)。

    与 process_task 同构：窗口聚类出的 segments -> cluster_local_speakers
    -> match_decision 三段式；仅有的差别是不落库、质心在内存里滚动。
    """
    import app.speakers as spk_mod
    from .service import cluster_local_speakers
    from .speakers import match_decision

    chunks = [p.stem for p in sorted((case / "chunks").glob("[0-9][0-9].m4a"))]
    offsets = json.loads((case / "cache" / "offsets.json").read_text())
    cache = _SpkEmbedCache(case, engine, window_t)
    centroids: list[tuple[np.ndarray, int]] = []
    decisions, utterances = [], []
    saved_margin = spk_mod.GRAY_MARGIN
    spk_mod.GRAY_MARGIN = gray_margin
    try:
        for idx, cname in enumerate(chunks):
            tokens, embs, starts, ends, _ = _load_cache(case, cname)
            segs = _segments_from_cache(tokens, embs, starts, ends, window_t)
            if not segs:
                continue
            spans_by_spk: dict[int, list] = {}
            for s in segs:
                spans_by_spk.setdefault(s.local_spk, []).append(
                    (s.start_ms, s.end_ms))
            spk_embs = cache.lookup(cname, spans_by_spk)
            emb_by_spans = {tuple(sorted(v)): k for k, v in spans_by_spk.items()}
            clusters = cluster_local_speakers(
                lambda spans: spk_embs[emb_by_spans[tuple(sorted(spans))]][0],
                segs, spk_t)
            pid_by_spk: dict[int, int] = {}
            for c in clusters:
                action, best, sim = match_decision(
                    centroids, c["emb"], spk_t, c["ms"], min_enroll_ms)
                if action == "merge":
                    cen, n = centroids[best]
                    centroids[best] = ((cen * n + c["emb"]) / (n + 1), n + 1)
                    pid = best
                elif action == "attach":
                    pid = best
                else:
                    centroids.append((c["emb"].copy(), 1))
                    pid = len(centroids) - 1
                for s_ in c["spks"]:
                    pid_by_spk[s_] = pid
                decisions.append({
                    "chunk": idx + 1, "ms": c["ms"], "action": action,
                    "person": pid, "best_sim": round(float(sim), 3),
                })
            for s_ in segs:
                utterances.append({
                    "chunk": idx + 1, "start_ms": offsets[idx] + s_.start_ms,
                    "end_ms": offsets[idx] + s_.end_ms,
                    "person": pid_by_spk[s_.local_spk], "text": s_.text,
                })
    finally:
        spk_mod.GRAY_MARGIN = saved_margin
    run = case / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in decisions))
    (run / "utterances.jsonl").write_text(
        "\n".join(json.dumps(u, ensure_ascii=False) for u in utterances))
    return decisions, utterances, len(centroids)


def load_truth(case: Path):
    p = case / "truth.jsonl"
    if not p.exists():
        return None
    out = []
    for line in p.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            r = json.loads(line)
            out.append((r["start_ms"], r["end_ms"], r["speaker"]))
    return out


def score(utterances, truth, total_ms: int):
    """帧级(10ms)说话人判定：匈牙利最优映射后报 DER 与逐人纯净度。"""
    from scipy.optimize import linear_sum_assignment
    step = 10
    n = max(total_ms // step, 1)
    t_lab = np.full(n, -1, dtype=int)
    h_lab = np.full(n, -1, dtype=int)
    t_names = sorted({s for _, _, s in truth})
    t_idx = {s: i for i, s in enumerate(t_names)}
    for a, b, s in truth:
        t_lab[a // step:min(b // step, n)] = t_idx[s]
    n_hyp = max((u["person"] for u in utterances), default=-1) + 1
    for u in utterances:
        h_lab[u["start_ms"] // step:min(u["end_ms"] // step, n)] = u["person"]
    if n_hyp == 0:
        return {"truth_speakers": len(t_names), "hyp_speakers": 0, "der": 1.0,
                "per_truth_purity": {s: 0.0 for s in t_names}}
    both = (t_lab >= 0) & (h_lab >= 0)
    conf = np.zeros((len(t_names), n_hyp), dtype=int)
    np.add.at(conf, (t_lab[both], h_lab[both]), 1)
    ri, ci = linear_sum_assignment(-conf)
    matched = int(conf[ri, ci].sum())
    speech = int((t_lab >= 0).sum())
    purity = {}
    for i, s in enumerate(t_names):
        tot = int(conf[i].sum())
        purity[s] = round(float(conf[i].max()) / tot, 3) if tot else 0.0
    return {
        "truth_speakers": len(t_names), "hyp_speakers": n_hyp,
        "der": round(1 - matched / speech, 3) if speech else 1.0,
        "hyp_only_frames": int(((t_lab < 0) & (h_lab >= 0)).sum()),
        "per_truth_purity": purity,
    }


def _engine_for(args):
    from .config import load_config
    from .main import _build_engine_
    return _build_engine_(load_config(args.config))


def cmd_replay(args):
    case = Path(args.case)
    engine = _engine_for(args)
    name = args.name or f"wt{args.window_t}_st{args.spk_t}_me{int(args.min_enroll_ms)}"
    decisions, utterances, n_persons = replay_case(
        case, engine, args.window_t, args.spk_t, args.min_enroll_ms,
        args.gray_margin, name)
    truth = load_truth(case)
    print(f"重放完成：{len(utterances)} 条 / 最终说话人 {n_persons} 个"
          f"（决策日志 runs/{name}/decisions.jsonl）")
    if truth:
        sc = score(utterances, truth, _wav_ms(case / "cache" / "full.wav"))
        print(json.dumps(sc, ensure_ascii=False, indent=1))
    else:
        print("（无 truth.jsonl 跳过打分；truth-draft 生成草稿后人工校对）")
    from collections import Counter
    spoken = Counter()
    for u in utterances:
        spoken[u["person"]] += u["end_ms"] - u["start_ms"]
    print("每人说话秒数:", {k: round(v / 1000, 1) for k, v in sorted(spoken.items())})


def cmd_sweep(args):
    case = Path(args.case)
    engine = _engine_for(args)
    truth = load_truth(case)
    total_ms = _wav_ms(case / "cache" / "full.wav")
    truth_n = len({s for _, _, s in truth}) if truth else None
    rows = []
    for wt in [float(x) for x in args.window_t.split(",")]:
        for st in [float(x) for x in args.spk_t.split(",")]:
            for me in [float(x) for x in args.min_enroll_ms.split(",")]:
                name = f"sw_wt{wt}_st{st}_me{int(me)}"
                _, utts, n = replay_case(case, engine, wt, st, me,
                                         args.gray_margin, name)
                sc = score(utts, truth, total_ms) if truth else None
                rows.append({"window_t": wt, "spk_t": st, "min_enroll_ms": me,
                             "persons": n, "der": sc and sc["der"],
                             "hit": truth_n is None or n == truth_n})
                print(f"wt={wt} st={st} me={int(me)} -> {n} 人"
                      + (f" DER {sc['der']}" if sc else ""))
    rows.sort(key=lambda r: (not r["hit"], r["der"] if r["der"] is not None else 9))
    lines = ["| window_t | spk_t | min_enroll_ms | 人数 | DER | 命中 |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['window_t']} | {r['spk_t']} | {int(r['min_enroll_ms'])} "
                     f"| {r['persons']} | {r['der']} | {'✓' if r['hit'] else ''} |")
    (case / "runs" / "sweep.md").write_text("\n".join(lines) + "\n")
    print(f"排序结果已写 {case / 'runs' / 'sweep.md'}")


def cmd_truth_draft(args):
    case = Path(args.case)
    tokens, embs, starts, ends, total_ms = _load_cache(case, "full")
    segs = _segments_from_cache(tokens, embs, starts, ends, args.window_t)
    if not segs:
        raise SystemExit("full 缓存无转写结果，先跑 prepare")
    spans_by_spk: dict[int, list] = {}
    for s in segs:
        spans_by_spk.setdefault(s.local_spk, []).append((s.start_ms, s.end_ms))
    ranked = sorted(spans_by_spk,
                    key=lambda k: -sum(e - b for b, e in spans_by_spk[k]))
    top = ranked[:3]
    label = {spk: chr(ord("A") + i) for i, spk in enumerate(top)}
    with (case / "truth.draft.jsonl").open("w") as f:
        for s in segs:
            if s.local_spk in label:
                f.write(json.dumps({
                    "start_ms": s.start_ms, "end_ms": s.end_ms,
                    "speaker": label[s.local_spk], "text": s.text},
                    ensure_ascii=False) + "\n")
    share = {label[k]: round(sum(e - b for b, e in spans_by_spk[k]) / 1000, 1)
             for k in top}
    print(f"草稿已写 {case / 'truth.draft.jsonl'}（window_t={args.window_t}，"
          f"取素材前 3 的簇）")
    print(f"各说话人素材秒数: {share}；音频总长 {total_ms / 1000:.1f}s")
    print("校对：按文本/听音频改 speaker 字段、删错段补漏段，"
          "另存为 truth.jsonl 生效")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", help="case 目录")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("replay")
    p.add_argument("--window-t", type=float, default=0.65)
    p.add_argument("--spk-t", type=float, default=0.45)
    p.add_argument("--min-enroll-ms", type=float, default=6000)
    p.add_argument("--gray-margin", type=float, default=0.10)
    p.add_argument("--name", default=None)
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("sweep")
    p.add_argument("--window-t", default="0.65")
    p.add_argument("--spk-t", default="0.45")
    p.add_argument("--min-enroll-ms", default="6000")
    p.add_argument("--gray-margin", type=float, default=0.10)
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("truth-draft")
    p.add_argument("--window-t", type=float, default=0.40)
    p.set_defaults(fn=cmd_truth_draft)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
