# nxcore-asr

非实时语音转写功能服务：FunASR（CPU）+ 会话内跨音频说话人识别 + 人工命名。
面向公司内多个 SaaS 后端，多租户共享 PostgreSQL（RLS 行级隔离）。

## 设计要点

- **两层权限模型**：身份只有"租户"（SaaS 调用方）一级。租户没有注册接口，只能由 config.yaml 静态配置（api_key + 名称）；启动时校验 key（拒绝占位符、短 key）与租户名（仅字母数字下划线连字符）。请求权限只凭 `Authorization: Bearer <key>` 判定，身份从 key 推导，绝不读请求参数。
- **会话属于唯一租户**：每个租户可注册多个会话；所有租户数据同表共存（每行带 `tenant_id`），靠 PostgreSQL 行级安全（RLS）隔离——应用以非超级用户 `asr_app` 连接，每个事务绑定租户上下文，数据库层按行过滤；跨租户资源一律 404，会话的删除因此只能由属主租户发起。忘了绑定上下文时一行都读不到（fail-closed）。
- **ORM**：SQLAlchemy 2.0。模型与 RLS 策略在 `app/db.py`，数据访问一律走 `tenant_session(engine, tenant)`。
- **会话**：由调用方显式注册（按 `external_id` 幂等）；永不自动删除，删除只能走接口，删了就全清（结果、说话人库、审计、文件）。
- **队列**：任务持久化在库里，进程重启自动恢复。同会话严格按上传顺序串行（认人有先后依赖），跨会话并行，全局并发 `max_workers`。
- **量化**：`llm_quantize: true` 时对转写模型的 LLM 分支做 int8 动态量化。逐字时间戳由模型内独立的 CTC 分支产出，说话人分离不受影响；开启后建议拿真实录音跑一次 `diarize_report` 校准。
- **认人**：cam++ 滑动窗口音色向量在单条音频内做平均链接聚类（阈值 `speaker_window_threshold`），结合 Fun-ASR-Nano 的逐字时间戳把每个字归到说话人、在切换点切分文本；音频内先按聚合音色合并碎编号，跨音频再与会话内各人质心比余弦，≥ `speaker_threshold` 归入并滑动更新质心，差 0.10 以内只挂靠不建卡（防幽灵声纹随录音数累积），明显不像才新建"说话人N"。亚秒级碎片只并入不新建人。阈值校准：拿真实录音跑 `python -m app.diarize_report <wav>` 看扫描表。
- **标签**：文本只存稳定 `person_id`，显示名读取时翻译——命名/合并后历史结果自动联动。每次标签变更会话 `version` +1，调用方发现版本变了就重拉。
- **音频**：转写成功即删原始音频；失败保留供 `retry`。日志不打转写文本与人名。

## 启动

```bash
docker compose up -d        # 起 PostgreSQL（127.0.0.1:25432，初始化 asr_app 角色与 RLS 所需权限）
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt   # 含 torch/funasr，约 2GB
# 编辑 config.yaml：用 openssl rand -hex 32 生成随机 api_key 填入两个租户
# （占位符 CHANGE_ME 和短 key 会被启动校验直接拒绝）
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8300
```

模型全部放在本地 `models/` 目录（已入 `.gitignore`，不入库）。**缺失时服务启动会自动从 modelscope 下载**（首次约 2.1GB，之后不再联网）；无网环境可从已部署机器直接拷贝整个 `models/` 目录。模型目录名与下载标识的对应关系见 `app/engine/funasr_engine.py` 顶部的 `MODELS`。
无网/测试环境可用假引擎：`ASR_FAKE_ENGINE=1 .venv/bin/uvicorn app.main:app ...`（无需模型）

## 内置测试台

服务启动后浏览器打开 `http://127.0.0.1:8300/console`：

1. 填入 `config.yaml` 里的 API key（保存在浏览器本地）
2. 填 `external_id` 注册会话
3. 点"开始录音"对着麦克风说话：每录满 1 分钟自动把这一段上传到当前会话并开始转写（同会话按顺序串行，说话人跨段连续），点按钮停止后剩余部分也会上传；或"上传文件"选本地音频
4. 转写结果实时显示；右侧每个说话人有"命名"按钮（已命名的可"改名"/"清除"），命名后历史结果联动；两个下拉选人可合并

多人测试：几个人轮流对麦克风讲话各录一段，观察跨音频是否认成同一人——这是验证 `speaker_threshold` 最直接的方式。页面本身无鉴权（接口调用仍需 key），只用于本机/内网调试。

## 接口

Base：`/v1`，全部需要 `Authorization: Bearer <api_key>`（health 也包括在内，统计只含本租户）。

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/asr-jobs` | 建转写任务，`{"external_id": "..."}`，按 external_id 幂等 get-or-create 会话；可带 `Idempotency-Key` 头（同会话同 key 返回已有 job，响应含当前 status） |
| POST | `/asr-jobs/{jid}/upload-authorization` | 换取直传地址，10 分钟有效；重新授权即作废旧地址 |
| PUT | `/uploads/{token}` | 直传音频裸字节（非 multipart）；complete 前可重复 PUT 覆盖；URL 由请求 Host 拼出，反代场景需保证 Host 头正确 |
| POST | `/asr-jobs/{jid}/upload-complete` | 确认上传并触发转写（分配 seq、入队）；对已入队/已完成的 job 幂等返回现状 |
| GET | `/asr-jobs/{jid}` | 轮询任务，与 `GET /tasks/{tid}` 响应完全相同（job_id 即 task_id） |
| POST | `/sessions` | 注册会话，`{"external_id": "..."}`，幂等（asr-jobs 的显式等价通道） |
| GET | `/sessions` | 会话列表带统计 |
| DELETE | `/sessions/{sid}` | 删除会话（全清），幂等；仅属主租户可删 |
| GET | `/tasks/{tid}` | 任务状态；done 时含 `utterances` |
| GET | `/sessions/{sid}/tasks` | 会话内任务列表（按 seq 倒序，limit 默认 50 上限 200） |
| GET | `/sessions/{sid}/results?after=0` | 按游标增量拉全会话结果，响应带 `session_version` |
| POST | `/tasks/{tid}/retry` | 重试失败任务（原始音频已删则提示重传） |
| GET | `/sessions/{sid}/speakers` | 说话人列表 |
| PUT | `/sessions/{sid}/speakers/{pid}/name` | `{"name": "张三"}`；name 为空即清除 |
| POST | `/sessions/{sid}/speakers/merge` | `{"from_person": "...", "into_person": "..."}`；两侧已有不同人工名则 409 |
| POST | `/sessions/{sid}/speakers/{pid}/unmerge` | 撤销合并 |
| GET | `/health` | 本租户任务统计 + 队列水位、引擎信息 |
| GET | `/console` | 内置测试台页面（静态页，接口调用仍需 key） |

上传走四步契约（与内部 SaaS 后端 asr-jobs 流程同构）：**建 job → 拿授权 → PUT 直传字节 → 确认入队**。任务状态机 `pending_upload → uploaded → queued → processing → done/failed`；建单未完成的 job 重启后留在库里，补调 upload-complete 即触发。同一 external_id 的多个 job 落同一会话，按 complete 先后串行转写，说话人跨段连续。

## 快速试用

```bash
KEY=$(grep '^    api_key:' config.yaml | head -1 | awk '{print $2}')   # 取第一个租户的 key
JOB=$(curl -s -X POST localhost:8300/v1/asr-jobs -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' -d '{"external_id":"demo-1"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
URL=$(curl -s -X POST localhost:8300/v1/asr-jobs/$JOB/upload-authorization -H "Authorization: Bearer $KEY" | python3 -c 'import sys,json;print(json.load(sys.stdin)["upload_url"])')
curl -s -X PUT "$URL" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/octet-stream' --data-binary @sample.wav
curl -s -X POST localhost:8300/v1/asr-jobs/$JOB/upload-complete -H "Authorization: Bearer $KEY"
curl -s localhost:8300/v1/asr-jobs/$JOB -H "Authorization: Bearer $KEY"   # 轮询到 status=done
```

## 测试

```bash
docker compose up -d   # 测试需要 PG 容器（自动建 asr_test 库）
.venv/bin/python -m pytest tests/   # 全部走假引擎，无需模型
```

## 已知边界

- 单机单进程；PostgreSQL + 进程内队列，多实例同时跑会重复消费任务（当前吞吐足够：1 分钟音频 CPU 转写约十几秒）。
- 说话人阈值 0.5 是起点：偏高会把同一人拆成多个，偏低会把不同人并成一个，务必拿真实业务音频标注一批验证后再调。
- 不做跨会话/跨租户的声纹身份复用——这是有意的边界，涉及生物识别合规，如需做应作为独立功能立项。
