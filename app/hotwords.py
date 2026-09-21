"""用户专属热词：转写后拼音同音纠偏（引擎无关）。

调用方（SaaS 侧）随任务提交该用户人物库的专有名词（姓名/别名/组织）。
FunASR 本地引擎（funasr / llama.cpp 两条路径）没有经典 paraformer 的
hotword 声学偏置能力，因此在转写落库前做后处理纠偏：转写文本里与热词
**拼音完全一致**的片段替换为热词——专有名词错字（同音/近形）恰好是
ASR 的主力错误形态（实测：汪工→王工、星桥剑社→星桥建设、
迁移站两周→迁移占两周类）。

保守取向：
- 只做**严格同音**替换（不带调拼音序列逐字相同）——近音模糊替换的误伤
  不可控，宁可漏纠不误纠；
- 热词长度 2~20 字才参与（单字同音替换误伤面太大：把所有「他/她」互换
  这类绝不是期望行为）；
- 长热词优先（先长后短，避免「星桥建设有限公司」被「建设」抢先半截）；
- 与热词相同的片段是 no-op，不重复替换、不重叠替换。

已知局限：
- 多音字按**候选片段本身**取音（lazy_pinyin 对 span 整段调用），不是按
  整段转写文本取音——位于片段边界、发音依赖外部上下文的多音字可能取错
  音。例如「银行」在片段内正确取 xíng，但孤立的「行」无法区分 xíng/háng。
- 等长且互为同音的热词（如「王工」/「汪工」）命中顺序由输入顺序决定：
  先出现的先匹配，行为确定。
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


@functools.lru_cache(maxsize=4096)
def _syllables(text: str) -> tuple[str, ...]:
    """不带调拼音序列。多音字按候选片段上下文取音（见模块级已知局限）。
    返回 tuple：不可变，防缓存值被外部意外修改。"""
    return tuple(lazy_pinyin(text, errors="ignore"))


def apply_hotwords(text: str, hotwords: list[str] | None) -> str:
    """对一段转写文本按热词表做同音纠偏，返回修正后的文本。"""
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
    words.sort(key=len, reverse=True)
    if not words:
        return text
    out: list[str] = []          # 已定稿的字符（含替换产物）
    i = 0
    n = len(text)
    while i < n:
        matched = False
        for w in words:
            k = len(w)
            if i + k > n:
                continue
            span = text[i:i + k]
            if span == w:
                out.append(w)
                i += k
                matched = True
                break
            # 音节数必须等于字符数——排除纯非汉字片段（英文/数字/符号的
            # 拼音序列为空元组 ()，() == () 会误匹配任意等长非汉字对）
            span_syl = _syllables(span)
            if len(span_syl) == k and span_syl == _syllables(w):
                out.append(w)
                i += k
                matched = True
                break
        if not matched:
            out.append(text[i])
            i += 1
    return "".join(out)
