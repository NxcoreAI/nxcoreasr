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
"""
from __future__ import annotations

from pypinyin import lazy_pinyin

MIN_WORD_LEN = 2
MAX_WORD_LEN = 20

_syllables_cache: dict[str, list[str]] = {}


def _syllables(text: str) -> list[str]:
    """不带调拼音序列（多音字按整段上下文取音）。小缓存：热词表逐任务复用。"""
    cached = _syllables_cache.get(text)
    if cached is None:
        cached = lazy_pinyin(text, errors="ignore")
        if len(_syllables_cache) > 4096:
            _syllables_cache.clear()
        _syllables_cache[text] = cached
    return cached


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
            # 逐字数需一致（拼音按整段取音以处理多音字）
            if len(span) == k and _syllables(span) == _syllables(w):
                out.append(w)
                i += k
                matched = True
                break
        if not matched:
            out.append(text[i])
            i += 1
    return "".join(out)
