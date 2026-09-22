"""从公开会议语料构建 diarize data case（取每场前 N 分钟 + rttm 转 truth）。

用法：
  .venv/bin/python scripts/build_meeting_case.py <src> <session_id> <case_dir> [--minutes 10]

src 目前支持：
  aishell4   HF: ggfox00000/dia-aishell4-test（真实会议 5-7 人，flac 单声道）
  alimeeting HF: ggfox00000/dia-alimeeting-test（真实会议 2-4 人，8 通道 wav 取第 1 通道）

产出 case_dir/{chunks/NN.m4a, truth.jsonl, notes.md}；truth 时间轴与分片拼接
严格对齐（prepare 会重建 full.wav 并按解码时长算偏移，与截取逻辑一致）。
"""
import argparse
import subprocess
import urllib.request
from pathlib import Path

MIRROR = "https://hf-mirror.com/datasets/ggfox00000"
LAYOUT = {
    "aishell4": {"audio": f"{MIRROR}/dia-aishell4-test/resolve/main/audio/test/%s.flac",
                 "rttm": f"{MIRROR}/dia-aishell4-test/resolve/main/rttm/test/%s.rttm",
                 "mb_per_min": 7.5},
    "alimeeting": {"audio": f"{MIRROR}/dia-alimeeting-test/resolve/main/audio/far/%s.wav",
                   "rttm": f"{MIRROR}/dia-alimeeting-test/resolve/main/rttm/far/%s.rttm",
                   "mb_per_min": 15.5},   # 8 通道；截取后 ffmpeg 取第 1 通道
}


def sh(*cmd):
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", choices=LAYOUT)
    ap.add_argument("session")
    ap.add_argument("case_dir")
    ap.add_argument("--minutes", type=int, default=10)
    a = ap.parse_args()
    lay = LAYOUT[a.src]
    case = Path(a.case_dir)
    (case / "chunks").mkdir(parents=True, exist_ok=True)

    # 只下载前 N 分钟的字节（aishell4 单场 250MB+、alimeeting 500MB，没必要全拉）
    part = case / "_source.part"
    need = int(lay["mb_per_min"] * (a.minutes + 1)) * 1024 * 1024
    req = urllib.request.Request(lay["audio"] % a.session,
                                 headers={"Range": f"bytes=0-{need}"})
    with urllib.request.urlopen(req, timeout=600) as r, part.open("wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    rttm_path = case / "_source.rttm"
    urllib.request.urlretrieve(lay["rttm"] % a.session, rttm_path)

    # 截 N 分钟 → 16k 单声道（alimeeting 8 通道取第 1 通道）
    wav = case / "full_src.wav"
    sh("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(part),
       "-t", str(a.minutes * 60),
       "-af", "pan=mono|c0=c0" if a.src == "alimeeting" else "aformat=channel_layouts=mono",
       "-ar", "16000", "-sample_fmt", "s16", str(wav))
    part.unlink()

    # rttm → truth（按截取时长过滤；毫秒）
    ms_cap = a.minutes * 60 * 1000
    spk_count = set()
    with (case / "truth.jsonl").open("w") as out:
        for line in rttm_path.read_text().splitlines():
            f = line.split()
            if len(f) >= 8 and f[0] == "SPEAKER":
                s, d, who = float(f[3]) * 1000, float(f[4]) * 1000, f[7]
                if s < ms_cap:
                    out.write('{"start_ms": %d, "end_ms": %d, "speaker": "%s"}\n'
                              % (s, min(s + d, ms_cap), who))
                    spk_count.add(who)
    rttm_path.unlink()

    # 15s 分片（模拟手机滚动上传；不对齐说话边界）
    import wave
    with wave.open(str(wav), "rb") as w:
        data, rate = w.readframes(w.getnframes()), w.getframerate()
    step_bytes = 15 * rate * 2  # 15s 的字节宽（16bit 单声道）
    n = 0
    for off in range(0, len(data), step_bytes):
        n += 1
        seg = data[off:off + step_bytes]
        tmp = case / f"chunks/{n:02d}.wav"
        with wave.open(str(tmp), "wb") as w2:
            w2.setnchannels(1); w2.setsampwidth(2); w2.setframerate(rate)
            w2.writeframes(seg)
        sh("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(tmp),
           "-c:a", "aac", "-b:a", "48k", str(tmp.with_suffix(".m4a")))
        tmp.unlink()
    wav.rename(case / "full.wav")  # 留整段参考；prepare 会重建对齐版
    print(f"{case.name}: {len(spk_count)} 人 / {a.minutes} 分钟 / {n} 片")


if __name__ == "__main__":
    main()
