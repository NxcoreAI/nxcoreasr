#!/usr/bin/env bash
# 准备 macOS 便携 PostgreSQL 二进制目录（打包版免 Docker 形态）。
# 用法：bash scripts/prepare-portable-pg-macos.sh [目标目录]
# 来源：Postgres.app 官方 universal binaries（免编译、免 brew 依赖树）。
set -euo pipefail

TARGET="${1:-pg-dist}"
VERSION="17.4"
DL_URL="https://postgresapp.com/postgresql-${VERSION}-runtime.zip"
TMPDIR_DL="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_DL"' EXIT

if [ "$(uname)" != "Darwin" ]; then
  echo "本脚本仅用于 macOS（Windows 用 EDB binaries zip，见 prepare-portable-pg-win）" >&2
  exit 1
fi

echo "下载 Postgres.app runtime ${VERSION}（universal，约 90MB）…"
curl -fL --retry 3 -o "$TMPDIR_DL/runtime.zip" "$DL_URL"

echo "解压到 ${TARGET}/ …"
rm -rf "$TARGET"
mkdir -p "$TARGET"
unzip -q "$TMPDIR_DL/runtime.zip" -d "$TMPDIR_DL/extracted"
# Postgres.app runtime zip 的结构：postgresql-{ver}/ 内是 bin/lib/share 等
SRC=$(find "$TMPDIR_DL/extracted" -maxdepth 2 -type d -name bin | head -1 | xargs dirname)
cp -R "$SRC/bin" "$SRC/lib" "$SRC/share" "$TARGET/"

# 瘦身：开发头文件与静态库不入包。
rm -rf "$TARGET/include" "$TARGET"/*/include 2>/dev/null || true
find "$TARGET" -name "*.a" -delete 2>/dev/null || true

echo "校验 initdb/pg_ctl …"
"$TARGET/bin/initdb" --version
"$TARGET/bin/pg_ctl" --version
echo "完成：${TARGET}（由 supervisor 的 portablePgDir() 自动探测）"
