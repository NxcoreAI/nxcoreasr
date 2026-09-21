"""热词同音纠偏测试：纯函数行为 + 端到端任务管线（fake 引擎）携带热词。"""
from app.hotwords import apply_hotwords


def test_same_sound_correction():
    # 严格同音替换：ASR 专有名词错字的典型形态
    assert apply_hotwords("汪工说方案定稿了", ["王工"]) == "王工说方案定稿了"
    assert apply_hotwords("星桥剑社的项目", ["星桥建设"]) == "星桥建设的项目"
    # 多个热词同时生效；长词优先不被短词截胡
    assert apply_hotwords("李芝洲在星桥剑社", ["李志舟", "星桥建设有限公司"]) == "李志舟在星桥剑社"
    # 已经正确的片段是 no-op
    assert apply_hotwords("王工在星桥建设", ["王工", "星桥建设"]) == "王工在星桥建设"


def test_non_chinese_no_false_positive():
    """P1 回归：纯非汉字片段（拼音序列为空元组）不触发误替换。"""
    assert apply_hotwords("请CD确认", ["AB"]) == "请CD确认"
    assert apply_hotwords("回了99", ["88"]) == "回了99"
    assert apply_hotwords("测试test数据", ["demo"]) == "测试test数据"
    # 混合中英：只有中文部分参与匹配
    assert apply_hotwords("汪工CD", ["王工"]) == "王工CD"
    # 中文+数字混合热词也安全（数字部分无拼音，音节数<字符数→不匹配）
    assert apply_hotwords("工号88号", ["工号99号"]) == "工号88号"


def test_conservative_no_correction():
    # 不同音不替换（宁漏纠不误纠）；不带调同音（望/王 wang）则替换——这是设计行为
    assert apply_hotwords("完全无关的内容", ["王工"]) == "完全无关的内容"
    assert apply_hotwords("黄工厂那边", ["王工"]) == "黄工厂那边"  # huang≠wang
    assert apply_hotwords("望工厂那边", ["王工"]) == "王工厂那边"  # wang=wang 同音
    # 单字/超长热词不参与（单字同音误伤面不可控）
    assert apply_hotwords("他来了", ["他"]) == "他来了"
    assert apply_hotwords("某内容", ["超" * 21]) == "某内容"
    # 空表/空文本安全
    assert apply_hotwords("任意", None) == "任意"
    assert apply_hotwords("", ["王工"]) == ""


def test_multichar_context_pinyin():
    # 多音字按候选片段上下文取音：「银行」的行=xing 而非 hang
    assert apply_hotwords("去银航办手续", ["银行"]) == "去银行办手续"


def test_pipeline_carries_hotwords(client):
    """端到端：建单带 hotwords → fake 引擎转写 → 纠偏管道携带。fake 引擎
    输出恒为正确文本（「你好，我是第一位说话人」），无法直接验证纠偏改变
    文本——纠偏正确性由单元测试覆盖；此用例验证字段透传与向后兼容。"""
    r = client.post("/v1/asr-jobs",
                    json={"external_id": "hotwords-e2e",
                          "hotwords": ["王工", "星桥建设"]},
                    headers={"Authorization": "Bearer test-key-a"})
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["created"] is True

    from tests.conftest import auth, make_wav, wait_done
    from urllib.parse import urlparse
    a = client.post(f"/v1/asr-jobs/{job['job_id']}/upload-authorization",
                    headers=auth()).json()
    client.put(urlparse(a["upload_url"]).path, content=make_wav(),
               headers={**auth(), "Content-Type": "application/octet-stream"})
    c = client.post(f"/v1/asr-jobs/{job['job_id']}/upload-complete",
                    headers=auth())
    assert c.status_code == 200
    done = wait_done(client, job["job_id"])
    assert done["status"] == "done"

    # 不带 hotwords 的字段也兼容（SaaS 老版本/开关关闭时）
    r2 = client.post("/v1/asr-jobs", json={"external_id": "hotwords-e2e"},
                     headers=auth())
    assert r2.status_code == 200
