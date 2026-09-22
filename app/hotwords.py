"""用户专属热词：转写后拼音同音纠偏（引擎无关）。

调用方（SaaS 侧）随任务提交该用户人物库的专有名词（姓名/别名/组织）。
FunASR 本地引擎（funasr / llama.cpp 两条路径）没有经典 paraformer 的
hotword 声学偏置能力，因此在转写落库前做后处理纠偏：转写文本里与热词
**拼音完全一致**的片段替换为热词——专有名词错字（同音/近形）恰好是
ASR 的主力错误形态（实测：汪工→王工、星桥剑社→星桥建设、
迁移站两周→迁移占两周类）。

保守取向（两档，2026-09-22）：
- 2 字热词：只做**严格同音**替换（不带调拼音逐字相同）——短词近音误伤面
  不可控（正总→郑总、同姓称谓互换），宁可漏纠不误纠；
- ≥3 字热词（人名全名/组织类）：放宽到**近音等价类**——平翘舌（zh↔z、
  ch↔c、sh↔s）、前后鼻音（n↔ng）、n↔l。等长三字近音碰撞概率远低于二字
  词，且词表本身是真实专有名词，误纠风险可控；
- 热词长度 2~20 字才参与（单字同音替换误伤面太大：把所有「他/她」互换
  这类绝不是期望行为）；
- 长热词优先（先长后短，避免「星桥建设有限公司」被「建设」抢先半截）；
- 与热词相同的片段是 no-op，不重复替换、不重叠替换。

匹配复杂度：O(N × L)，N=文本长度、L=热词长度种类数（≤19）——每个位置
对每种长度只算一次窗口拼音，音节匹配用字典查找。替代朴素 O(N × W × k)。

已知局限：
- 多音字按**候选片段本身**取音（lazy_pinyin 对 span 整段调用），不是按
  整段转写文本取音——位于片段边界、发音依赖外部上下文的多音字可能取错
  音。例如「银行」在片段内正确取 xíng，但孤立的「行」无法区分 xíng/háng。
- 等长且互为同音的热词（如「王工」/「汪工」）命中顺序由输入顺序决定：
  先出现的先匹配（构建查找表时后者覆盖前者），行为确定。
- seg.text 被替换后，若引擎输出含词级时间戳，词与文本对齐关系可能失效
  （当前引擎无此依赖，Utterance 时间戳来自 segment 级而非词级）。
"""
from __future__ import annotations

import functools
import logging

from pypinyin import lazy_pinyin

log = logging.getLogger(__name__)

MIN_WORD_LEN = 2
MAX_WORD_LEN = 20
# ≥3 字热词的近音档阈值（2 字保持严格同音）
NEAR_PHONE_MIN_LEN = 3

# 近音等价类（≥3 字热词）：平翘舌 + n/l。键按「长组合在前」处理（zh 在 z 前）。
_INITIAL_CLASS = {"zh": "z", "ch": "c", "sh": "s", "l": "n"}


@functools.lru_cache(maxsize=4096)
def _syllables(text: str) -> tuple[str, ...]:
    """不带调拼音序列。多音字按候选片段上下文取音（见模块级已知局限）。
    返回 tuple：不可变，防缓存值被外部意外修改。"""
    return tuple(lazy_pinyin(text, errors="ignore"))


def _canon_syllable(syl: str) -> str:
    """音节 → 近音等价类代表：平翘舌归一、n/l 归一、后鼻音去 g（n↔ng）。"""
    for src, dst in _INITIAL_CLASS.items():
        if syl.startswith(src):
            syl = dst + syl[len(src):]
            break
    if syl.endswith("ng"):
        syl = syl[:-1]
    return syl


def _build_lookup(words: list[str]) -> tuple[set[str], dict[int, dict[tuple[str, ...], str]], list[int]]:
    """热词表 → 索引结构：精确命中集 + 按长度分组的拼音查找表 + 长度降序列表。

    查找键按档位（2026-09-22）：2 字=严格同音序列；≥3 字（人名全名/组织类）
    =近音等价类序列（严格同音是其子集，归一后一张表即可）。同长度且同键的
    热词：后加入的覆盖先加入的（dict 覆盖语义）——等长同音场景由输入顺序
    决定优先级，行为确定（已在模块 docstring 文档化）。
    """
    exact: set[str] = set()
    by_len: dict[int, dict[tuple[str, ...], str]] = {}
    for w in words:
        exact.add(w)
        w_syl = _syllables(w)
        # 音节数≠字符数的热词（含非汉字）跳过查找表——同音匹配无意义
        if len(w_syl) == len(w):
            key = tuple(_canon_syllable(s) for s in w_syl) if len(w) >= NEAR_PHONE_MIN_LEN else w_syl
            by_len.setdefault(len(w), {})[key] = w
    return exact, by_len, sorted(by_len.keys(), reverse=True)


def apply_hotwords(text: str, hotwords: list[str] | None) -> str:
    """对一段转写文本按热词表做拼音纠偏（2 字严格同音/≥3 字近音等价类）。"""
    if not text or not hotwords:
        return text
    # 过滤+去重+按长度降序（长词优先，防短词截胡长词的局部）
    words: list[str] = []
    seen: set[str] = set()
    for raw in hotwords:
        w = (raw or "").strip()
        if MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN and w not in seen:
            seen.add(w)
            words.append(w)
        elif w and w not in seen:
            # 可观测性：无效热词静默丢弃是工单排查的盲区
            log.warning("热词被过滤（长度 %d 超出 [%d,%d] 或为空）: %r",
                        len(w), MIN_WORD_LEN, MAX_WORD_LEN, w[:20])
            seen.add(w)
    if not words:
        return text
    exact, by_len, lengths = _build_lookup(words)
    out: list[str] = []          # 已定稿的字符（含替换产物）
    i = 0
    n = len(text)
    while i < n:
        matched = False
        for k in lengths:        # ≤19 种长度，降序
            if i + k > n:
                continue
            span = text[i:i + k]
            if span in exact:    # 精确命中快路径（免拼音计算）
                out.append(span)
                i += k
                matched = True
                break
            # 音节数必须等于字符数——排除纯非汉字片段（英文/数字/符号的
            # 拼音序列为空元组 ()，() == () 会误匹配任意等长非汉字对）
            span_syl = _syllables(span)
            if len(span_syl) == k:
                key = tuple(_canon_syllable(s) for s in span_syl) if k >= NEAR_PHONE_MIN_LEN else span_syl
                w = by_len[k].get(key)
                if w:
                    out.append(w)
                    i += k
                    matched = True
                    break
        if not matched:
            out.append(text[i])
            i += 1
    return "".join(out)
