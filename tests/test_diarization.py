"""说话人窗口聚类与切分纯函数的单元测试（不加载模型）。"""
import numpy as np

from app.engine.funasr_engine import (
    cluster_windows,
    group_segments,
    merge_short_fragments,
    slide_windows,
    smooth_token_runs,
    smooth_turns,
    token_speakers,
)
from app.engine.base import Segment


def _emb(angle_deg: float) -> np.ndarray:
    r = np.deg2rad(angle_deg)
    return np.array([np.cos(r), np.sin(r)], dtype=np.float32)


def tok(ch, start, end):
    return {"token": ch, "start": start, "end": end}


def seg(text, start, end, spk):
    return Segment(start_ms=start, end_ms=end, text=text, local_spk=spk)


class TestMergeShortFragments:
    def test_head_fragment_joins_next(self):
        # 新一句话头 60ms 碎片，距后段更近 → 并入后段
        out = merge_short_fragments([seg("提出来。", 180, 6300, 0),
                                     seg("我", 6600, 6660, 0),
                                     seg("觉得", 6720, 9000, 1)])
        assert len(out) == 2
        assert out[1].text == "我觉得"
        assert out[1].start_ms == 6600 and out[1].local_spk == 1

    def test_mid_fragment_joins_nearest(self):
        out = merge_short_fragments([seg("保证。", 7140, 12660, 1),
                                     seg("一周可", 12960, 13500, 1),
                                     seg("能太长", 13560, 18000, 0)])
        # 距前段间隙 300ms、距后段 60ms → 并入后段并改随其说话人
        assert len(out) == 2
        assert out[1].text == "一周可能太长"
        assert out[1].local_spk == 0

    def test_long_segments_untouched(self):
        segs = [seg("你好。", 0, 1500, 0), seg("再见。", 2000, 3600, 1)]
        assert merge_short_fragments(segs) == segs

    def test_single_short_segment_kept(self):
        segs = [seg("嗯。", 0, 300, 0)]
        assert merge_short_fragments(segs) == segs


class TestSlideWindows:
    def test_short_region_single_window(self):
        assert slide_windows([(0, 3000)]) == [(0, 3000)]

    def test_long_region_steps_by_hop(self):
        wins = slide_windows([(0, 5000)])
        assert wins[0] == (0, 3000)
        assert wins[1] == (1000, 4000)
        assert wins[-1] == (2000, 5000)  # 尾部向前借齐成整窗
        assert len(wins) == 3

    def test_region_slightly_over_window(self):
        # 只多出半秒：一个正窗 + 一个借齐的尾窗
        assert slide_windows([(0, 3500)]) == [(0, 3000), (500, 3500)]

    def test_empty_region_skipped(self):
        assert slide_windows([(100, 100)]) == []

    def test_multiple_regions(self):
        wins = slide_windows([(0, 1600), (5000, 8500)])
        assert wins == [(0, 1600), (5000, 8000), (5500, 8500)]


class TestClusterWindows:
    def test_two_clusters(self):
        # 0°/10° 同人，180°/190° 另一人
        embs = np.stack([_emb(0), _emb(10), _emb(180), _emb(190), _emb(5)])
        assert cluster_windows(embs, 0.95) == [0, 0, 1, 1, 0]

    def test_all_same(self):
        embs = np.stack([_emb(0), _emb(2), _emb(1)])
        assert cluster_windows(embs, 0.9) == [0, 0, 0]

    def test_empty_and_single(self):
        assert cluster_windows(np.zeros((0, 2), dtype=np.float32), 0.9) == []
        assert cluster_windows(np.stack([_emb(0)]), 0.9) == [0]

    def test_labels_ordered_by_first_appearance(self):
        embs = np.stack([_emb(180), _emb(190), _emb(0), _emb(5)])
        assert cluster_windows(embs, 0.9) == [0, 0, 1, 1]


class TestSmoothTurns:
    def test_short_blip_absorbed(self):
        # 说话人 1 只闪现 0.5s，应并入更长的邻段
        turns = [[0, 5000, 0], [5500, 6000, 1], [6000, 12000, 0]]
        out = smooth_turns(turns)
        assert len(out) == 1
        assert out[0] == [0, 12000, 0]

    def test_long_turn_kept(self):
        turns = [[0, 5000, 0], [5500, 7000, 1], [7500, 12000, 0]]
        assert smooth_turns(turns) == turns

    def test_alternating_blips_converge(self):
        turns = [[0, 5000, 0], [5500, 6000, 1], [6000, 6400, 0], [6400, 6900, 1]]
        out = smooth_turns(turns)
        assert len(out) == 1

    def test_single_turn(self):
        assert smooth_turns([[0, 300, 3]]) == [[0, 300, 3]]


class TestTokenSpeakers:
    def test_overlap_majority(self):
        turns = [[0, 5000, 0], [5000, 10000, 1]]
        tokens = [tok("a", 100, 200), tok("b", 5100, 5200)]
        assert token_speakers(tokens, turns) == [0, 1]

    def test_no_overlap_falls_back_to_previous(self):
        turns = [[0, 5000, 0], [6000, 10000, 1]]
        tokens = [tok("a", 100, 200), tok("b", 5500, 5600), tok("c", 6100, 6200)]
        assert token_speakers(tokens, turns) == [0, 0, 1]

    def test_no_turns(self):
        assert token_speakers([tok("a", 0, 100)], []) == [0]

    def test_punctuation_follows_previous_word(self):
        turns = [[0, 5000, 0], [5000, 10000, 1]]
        tokens = [tok("好", 4800, 4900), tok("。", 4900, 4960), tok("嗯", 5100, 5200)]
        # 句号落在说话人 1 的片段里，但应跟随"好"归说话人 0
        assert token_speakers(tokens, turns) == [0, 0, 1]

    def test_leading_punctuation_follows_next(self):
        turns = [[0, 5000, 0]]
        tokens = [tok("，", 0, 60), tok("好", 100, 200)]
        assert token_speakers(tokens, turns) == [0, 0]


class TestSmoothTokenRuns:
    def test_short_run_absorbed_to_nearer_side(self):
        # 中间 200ms 的 b：距前侧间隙 100ms，距后侧 10ms → 并入后侧
        tokens = [tok("a", 0, 1000), tok("b", 1100, 1300), tok("c", 1310, 2400)]
        assert smooth_token_runs(tokens, [0, 1, 2]) == [0, 2, 2]

    def test_long_run_kept(self):
        tokens = [tok("a", 0, 1000), tok("b", 1100, 2100), tok("c", 2200, 3200)]
        assert smooth_token_runs(tokens, [0, 1, 0]) == [0, 1, 0]

    def test_absorbed_into_same_speaker_neighbor_merges(self):
        tokens = [tok("a", 0, 1000), tok("b", 1100, 1300), tok("c", 1400, 2400)]
        assert smooth_token_runs(tokens, [0, 1, 0]) == [0, 0, 0]

    def test_tie_gaps_absorbed_forward(self):
        # 两侧间隙相等 → 归后侧
        tokens = [tok("a", 0, 1000), tok("b", 1100, 1300), tok("c", 1400, 2400)]
        assert smooth_token_runs(tokens, [0, 1, 2]) == [0, 2, 2]


class TestGroupSegments:
    def test_speaker_change_splits(self):
        tokens = [tok("你", 0, 200), tok("好", 200, 400), tok("再", 500, 700), tok("见", 700, 900)]
        segs = group_segments(tokens, [0, 0, 1, 1])
        assert len(segs) == 2
        assert segs[0].text == "你好" and segs[0].local_spk == 0
        assert segs[1].text == "再见" and segs[1].local_spk == 1

    def test_sentence_enders_split_within_speaker(self):
        tokens = [tok("好", 0, 100), tok("。", 100, 150), tok("行", 200, 300)]
        segs = group_segments(tokens, [0, 0, 0])
        assert [s.text for s in segs] == ["好。", "行"]

    def test_comma_does_not_split(self):
        tokens = [tok("好", 0, 100), tok("，", 100, 150), tok("行", 200, 300)]
        segs = group_segments(tokens, [0, 0, 0])
        assert len(segs) == 1

    def test_ender_kept_with_its_sentence(self):
        tokens = [tok("好", 0, 100), tok("。", 100, 150), tok("再", 200, 300)]
        segs = group_segments(tokens, [0, 0, 1])
        assert [s.text for s in segs] == ["好。", "再"]
        assert segs[0].end_ms == 150

    def test_empty(self):
        assert group_segments([], []) == []
