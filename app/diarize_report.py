"""说话人分离诊断：对单个 16k 单声道 wav 输出阈值扫描表与说话人相似度矩阵。

用法：python -m app.diarize_report <wav> [--config config.yaml]

拿到真实多人录音后跑一遍：扫描表里"簇数 = 真实人数"对应的窗口阈值、
相似度矩阵里各人之间的间隔，就是 config.yaml 两个 speaker 阈值该有的值。
窗口向量同时落盘 <wav>.emb.npz，换阈值反复分析不必重跑模型。
"""
import argparse
import wave

import numpy as np

from .engine.base import normalize
from .engine.funasr_engine import SPK_BATCH, cluster_windows, slide_windows


def _window_embeddings(engine, wav: str):
    vr = engine.vad.generate(input=wav)[0]
    regions = [(int(a), int(b)) for a, b in vr.get("value") or []]
    wins = slide_windows(regions)
    pcm, rate = engine._load_pcm(wav)
    slices = [pcm[a * rate // 1000: b * rate // 1000] for a, b in wins]
    embs = []
    for j in range(0, len(slices), SPK_BATCH):
        for r in engine.spk.generate(input=slices[j:j + SPK_BATCH], fs=rate):
            e = r["spk_embedding"]
            if hasattr(e, "detach"):
                e = e.detach().cpu().numpy()
            embs.append(normalize(np.asarray(e).reshape(-1)))
    return regions, wins, np.stack(embs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav", help="16k 单声道 wav 路径")
    ap.add_argument("--config", default=None, help="config.yaml 路径（默认取 ASR_CONFIG 或 ./config.yaml）")
    args = ap.parse_args()

    from .config import load_config
    from .engine.funasr_engine import FunasrEngine

    cfg = load_config(args.config)
    engine = FunasrEngine(cfg.model_dir, cfg.speaker_window_threshold)

    with wave.open(args.wav, "rb") as w:
        dur_ms = int(w.getnframes() / w.getframerate() * 1000)
    regions, wins, embs = _window_embeddings(engine, args.wav)
    speech_ms = sum(b - a for a, b in regions)
    if not wins:
        print(f"音频 {dur_ms / 1000:.1f}s，未检测到语音，结束")
        return
    print(f"音频 {dur_ms / 1000:.1f}s，语音 {speech_ms / 1000:.1f}s（{len(regions)} 段），"
          f"窗口 {len(wins)} 个（{wins[0][1] - wins[0][0]}ms 窗）")

    print("\n窗口聚类阈值扫描（簇数应≈真实人数）：")
    for t in [round(0.35 + 0.05 * i, 2) for i in range(8)]:
        n = len(set(cluster_windows(embs, t)))
        mark = "  ← 当前配置" if abs(t - cfg.speaker_window_threshold) < 1e-9 else ""
        print(f"  阈值 {t:.2f} → {n} 个说话人{mark}")

    np.savez(f"{args.wav}.emb.npz", embs=embs,
             starts=np.array([a for a, _ in wins]), ends=np.array([b for _, b in wins]))
    print(f"\n窗口向量已落盘 {args.wav}.emb.npz（{embs.shape}）")

    segments = engine.transcribe(args.wav)
    spans_by_spk: dict[int, list] = {}
    for s in segments:
        spans_by_spk.setdefault(s.local_spk, []).append((s.start_ms, s.end_ms))
    if not spans_by_spk:
        print("未检测到语音，结束")
        return
    print(f"\n按当前配置转写：{len(segments)} 段，音频内 {len(spans_by_spk)} 个说话人")
    ids = sorted(spans_by_spk)
    agg = {k: engine.embed(args.wav, spans_by_spk[k]) for k in ids}
    for k in ids:
        ms = sum(e - b for b, e in spans_by_spk[k])
        print(f"  说话人{k}: {len(spans_by_spk[k])} 段 / {ms / 1000:.1f}s")
    print("\n说话人间聚合向量相似度（同人跨段应明显高于不同人；对角线为 1）：")
    print("      " + "".join(f"{k:>8d}" for k in ids))
    for r in ids:
        row = "".join(f"{float(np.dot(agg[r], agg[c])):>8.2f}" for c in ids)
        print(f"{r:>5d} {row}")


if __name__ == "__main__":
    main()
