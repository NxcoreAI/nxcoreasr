-- 应用连接角色：非超级用户（超商会绕过 RLS）。
-- 拥有 asr 库以便建表；表上 FORCE ROW LEVEL SECURITY 让 owner 也受策略约束。
CREATE ROLE asr_app LOGIN PASSWORD 'asr_dev_password';
ALTER DATABASE asr OWNER TO asr_app;
