"""说话人库：跨音频认人、命名、合并（别名指针）、审计。

约定：utterances 只存 person_id；merged_into 形成别名链，读取时
resolve 到根，显示名取根的 display_name（无则 auto_label）。
"""
import re
import secrets

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db import LabelAudit, Person, SessionRecord, Task, Utterance, now_iso
from .engine.base import normalize


def _new_person_id() -> str:
    return "spk_" + secrets.token_urlsafe(12)


def bump_version(db: Session, session_id: str) -> None:
    rec = db.get(SessionRecord, session_id)
    if rec is not None:
        rec.version += 1


def audit(db: Session, session_id: str, person_id: str, action: str,
          old_value: str | None, new_value: str | None):
    db.add(LabelAudit(session_id=session_id, person_id=person_id, action=action,
                      old_value=old_value, new_value=new_value, at=now_iso()))


def resolve_root(persons_by_id: dict, person_id: str) -> Person:
    """沿 merged_into 链找到根。链由代码保证无环（merge 前先 resolve 双方）。"""
    p = persons_by_id[person_id]
    seen = set()
    while p.merged_into and p.id not in seen:
        seen.add(p.id)
        p = persons_by_id[p.merged_into]
    return p


def load_persons(db: Session, session_id: str) -> dict[str, Person]:
    rows = db.scalars(select(Person).where(Person.session_id == session_id)).all()
    return {p.id: p for p in rows}


def display_name_of(root: Person) -> str:
    return root.display_name or root.auto_label


def name_map(db: Session, session_id: str) -> dict[str, dict]:
    """person_id → 根人的 {person_id, name, named}，供结果读取时翻译。"""
    persons = load_persons(db, session_id)
    out = {}
    for pid in persons:
        root = resolve_root(persons, pid)
        out[pid] = {
            "person_id": root.id,
            "person_name": display_name_of(root),
            "named": root.display_name is not None,
        }
    return out


GRAY_MARGIN = 0.10  # 擦边区宽度：最佳相似度落在 [阈值-0.10, 阈值) 时只挂靠不建卡。
                     # 真实远场音频里同人跨录音相似度常落在这个区间，直接新建会随录音数累积幽灵

ATTACH_LEARN = 0.3  # 擦边挂靠的低权重质心学习率。完全不学会让远场相近嗓子永远
                     # 停在灰区：每一片都重新押注、押歪了跌破建卡线就开新卡（实测
                     # 一晚真实使用攒出 11 张碎片卡）；低权重既让库逐步收敛，一次
                     # 挂错也带不偏质心。embed_count 只计整权向量，不随挂靠增长。

ATTACH_LEARN_ENABLED = False  # 实验性挂靠学习总开关（见 match_or_create 内注释）

ATTACH_LEARN_MARGIN = 0.10  # 学习的无歧义门控：最佳相似度须领先第二名这么多才学。
                     # 近场相近异人的两张卡互相纠缠（领先 <0.1），学了会把两卡质心
                     # 拉到一起、毁掉阈值辛苦建立的分离；远场碎片卡对主卡领先 0.3+，
                     # 恰是需要学习收敛的场景。


def match_decision(centroids: list[tuple[np.ndarray, int]], emb: np.ndarray,
                   threshold: float, ms: float,
                   min_enroll_ms: float) -> tuple[str, int, float, float]:
    """三段式认人判定（纯函数，线上 match_or_create 与 diarize_eval 重放共用）。

    centroids 为各根人的 (质心未归一化均值, 向量数)；emb 为该簇聚合音色向量。
    返回 (action, best_idx, best_sim, second_sim)：action=merge 归入并应更新
    质心、attach 只挂靠不动质心、new 新建说话人。空库保底新建（首段音频
    锚点）。second_sim 是次佳相似度，供挂靠学习的无歧义门控用。
    """
    emb = normalize(emb)
    best_idx, best_sim, second_sim = -1, -1.0, -1.0
    for i, (c, n) in enumerate(centroids):
        sim = float(np.dot(emb, normalize(np.asarray(c, dtype=np.float32))))
        if sim > best_sim:
            second_sim, best_sim, best_idx = best_sim, sim, i
        elif sim > second_sim:
            second_sim = sim
    if best_idx >= 0 and best_sim >= threshold:
        return "merge", best_idx, best_sim, second_sim
    if best_idx >= 0 and (ms < min_enroll_ms or best_sim >= threshold - GRAY_MARGIN):
        return "attach", best_idx, best_sim, second_sim
    return "new", best_idx, best_sim, second_sim


def match_or_create(db: Session, session_id: str, emb: np.ndarray, threshold: float,
                    ms: float, min_enroll_ms: float) -> str:
    """与会话内各根人的质心比余弦；达标归入并滑动更新质心。

    判定走 match_decision；merge 归入并整权更新质心；attach 只挂最像的
    已有根人（不新建），默认不动质心（实验性低权重学习见 ATTACH_LEARN_
    ENABLED）；new 才是真正的新说话人。

    centroid 存的是各成员向量的真实均值（未归一化），比较时归一，
    这样滑动更新 (均值*n + 新向量)/(n+1) 才是精确的。"""
    persons = load_persons(db, session_id)
    roots = [(normalize(np.frombuffer(p.centroid, dtype=np.float32)), p.embed_count)
             for p in persons.values() if not p.merged_into and p.centroid]
    action, idx, best_sim, second_sim = match_decision(
        roots, emb, threshold, ms, min_enroll_ms)
    if action != "new":
        best = [p for p in persons.values() if not p.merged_into and p.centroid][idx]
    if action == "merge":
        n = best.embed_count
        centroid = (np.frombuffer(best.centroid, dtype=np.float32) * n
                    + normalize(emb)) / (n + 1)
        best.centroid = np.asarray(centroid, dtype=np.float32).tobytes()
        best.embed_count = n + 1
        return best.id
    if action == "attach":
        if ATTACH_LEARN_ENABLED and best_sim - second_sim >= ATTACH_LEARN_MARGIN:
            # 实验性（默认关）：无歧义挂靠低权重学习。远场灰区碎片靠它收敛，
            # 但近场相近异人（互相纠缠 <0.1 领先）稳态会被打破——四个 data case
            # 回归确认任何门控都救不了，留开关等远场 truth 标注后再评估
            n = best.embed_count
            centroid = (np.frombuffer(best.centroid, dtype=np.float32) * n
                        + ATTACH_LEARN * normalize(emb)) / (n + ATTACH_LEARN)
            best.centroid = np.asarray(centroid, dtype=np.float32).tobytes()
        return best.id
    root_count = len(roots)
    pid = _new_person_id()
    db.add(Person(id=pid, session_id=session_id, auto_label=f"说话人{root_count + 1}",
                  centroid=np.asarray(normalize(emb), dtype=np.float32).tobytes(),
                  embed_count=1))
    return pid


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def sanitize_name(name: str | None) -> str | None:
    if name is None:
        return None
    name = _CONTROL_CHARS.sub("", name).strip()
    if not name:
        return None
    if len(name) > 64:
        raise ValueError("名字过长（最多 64 字符）")
    return name


class SpeakerError(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail


def set_name(db: Session, session_id: str, person_id: str, name: str | None):
    persons = load_persons(db, session_id)
    if person_id not in persons:
        raise SpeakerError(404, "说话人不存在")
    root = resolve_root(persons, person_id)
    old = root.display_name
    root.display_name = name
    audit(db, session_id, root.id, "name" if name else "clear", old, name)
    bump_version(db, session_id)


def merge(db: Session, session_id: str, from_id: str, into_id: str):
    persons = load_persons(db, session_id)
    if from_id not in persons or into_id not in persons:
        raise SpeakerError(404, "说话人不存在")
    from_root = resolve_root(persons, from_id)
    into_root = resolve_root(persons, into_id)
    if from_root.id == into_root.id:
        return  # 已是同一人，幂等
    fn, tn = from_root.display_name, into_root.display_name
    if fn and tn and fn != tn:
        raise SpeakerError(409, f"两个说话人已有不同人工命名（{fn} / {tn}），请先清除一侧再合并")
    if fn and not tn:
        into_root.display_name = fn
        audit(db, session_id, into_root.id, "name", tn, fn)
    from_root.merged_into = into_root.id
    audit(db, session_id, from_root.id, "merge", None, into_root.id)
    bump_version(db, session_id)


def unmerge(db: Session, session_id: str, person_id: str):
    persons = load_persons(db, session_id)
    if person_id not in persons:
        raise SpeakerError(404, "说话人不存在")
    p = persons[person_id]
    if not p.merged_into:
        raise SpeakerError(409, "该说话人未被合并")
    old_target = p.merged_into
    p.merged_into = None
    audit(db, session_id, person_id, "unmerge", old_target, None)
    bump_version(db, session_id)


def list_speakers(db: Session, session_id: str) -> list[dict]:
    persons = load_persons(db, session_id)
    stats: dict[str, dict] = {}
    rows = db.execute(
        select(Utterance.person_id, Utterance.task_id).distinct()
        .where(Utterance.session_id == session_id)
    ).all()
    counts = db.execute(
        select(Utterance.person_id, func.count())
        .where(Utterance.session_id == session_id)
        .group_by(Utterance.person_id)
    ).all()
    for pid, n in counts:
        if pid not in persons:
            continue
        root = resolve_root(persons, pid)
        s = stats.setdefault(root.id, {"utterances": 0, "audio_tasks": set()})
        s["utterances"] += n
    for pid, task_id in rows:
        if pid not in persons:
            continue
        root = resolve_root(persons, pid)
        s = stats.setdefault(root.id, {"utterances": 0, "audio_tasks": set()})
        s["audio_tasks"].add(task_id)
    out = []
    for p in persons.values():
        if p.merged_into:
            continue
        s = stats.get(p.id, {"utterances": 0, "audio_tasks": set()})
        merged_children = [q.id for q in persons.values() if q.merged_into == p.id]
        out.append({
            "person_id": p.id,
            "auto_label": p.auto_label,
            "display_name": p.display_name,
            "named": p.display_name is not None,
            "utterance_count": s["utterances"],
            "audio_count": len(s["audio_tasks"]),
            "merged_person_ids": merged_children,
        })
    out.sort(key=lambda x: x["auto_label"])
    return out


def speaker_tasks(db: Session, session_id: str, person_id: str) -> dict:
    """某说话人出现过的录音列表：按根人归拢，带每次录音内的话句数与说话时长。"""
    persons = load_persons(db, session_id)
    if person_id not in persons:
        raise SpeakerError(404, "说话人不存在")
    root = resolve_root(persons, person_id)
    pids = [q.id for q in persons.values() if resolve_root(persons, q.id).id == root.id]
    rows = db.execute(
        select(Utterance.task_id, func.count(),
               func.sum(Utterance.end_ms - Utterance.start_ms))
        .where(Utterance.session_id == session_id, Utterance.person_id.in_(pids))
        .group_by(Utterance.task_id)
    ).all()
    task_ids = [r[0] for r in rows]
    tasks = ({t.id: t for t in db.scalars(
        select(Task).where(Task.id.in_(task_ids))).all()} if task_ids else {})
    items = []
    for tid, cnt, spoken in rows:
        t = tasks.get(tid)
        items.append({
            "task_id": tid,
            "seq": t.seq if t else None,
            "duration_ms": t.duration_ms if t else None,
            "created_at": t.created_at if t else None,
            "done_at": t.done_at if t else None,
            "utterance_count": cnt,
            "spoken_ms": int(spoken or 0),
        })
    items.sort(key=lambda x: (x["seq"] is None, x["seq"], x["task_id"]))
    return {"person_id": root.id, "person_name": display_name_of(root),
            "named": root.display_name is not None, "tasks": items}
