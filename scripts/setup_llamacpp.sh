#!/usr/bin/env bash
# 一键准备 llama.cpp 引擎所需的资产：量化模型（GGUF）+ 补丁版 llama-funasr-cli 二进制。
#   用法：bash scripts/setup_llamacpp.sh [目标目录]
#   默认装到 models/llamacpp/（models/ 已在 .gitignore，不入库）：
#     bin/llama-funasr-cli   带 --dump-enc 落盘补丁的自编译二进制
#     gguf/*.gguf            编码器 + LLM 量化 + VAD 三个模型文件（约 1.7G）
#   依赖：git、C 编译器、cmake（没有的话会用当前 python 环境装一个）。
#   下载走 hf-mirror.com（国内快），可用 HF_HOST 环境变量覆盖；拉不到时自动换 huggingface.co。
set -euo pipefail

DEST="${1:-models/llamacpp}"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|AMD64) LLM="qwen3-0.6b-q4km.gguf" ;;   # 老至强 x86：q4 明显快于 q8
  arm64|aarch64) LLM="qwen3-0.6b-q8_0.gguf" ;;  # Apple 芯片：q8 与 q4 同速、文本更准
  *) echo "未知架构 $ARCH"; exit 1 ;;
esac

GGUF_FILES=(funasr-encoder-f16.gguf "$LLM" fsmn-vad.gguf)
GGUF_REPOS=(FunAudioLLM/Fun-ASR-Nano-GGUF FunAudioLLM/Fun-ASR-Nano-GGUF FunAudioLLM/fsmn-vad-GGUF)

BIN_DIR="$DEST/bin"; GGUF_DIR="$DEST/gguf"
mkdir -p "$BIN_DIR" "$GGUF_DIR"

# ---- 1. 下载 GGUF（已存在且非空的跳过）----
HF_HOSTS=("${HF_HOST:-https://hf-mirror.com}" "https://huggingface.co")
for i in "${!GGUF_FILES[@]}"; do
  f="${GGUF_FILES[$i]}"; repo="${GGUF_REPOS[$i]}"
  out="$GGUF_DIR/$f"
  if [[ -s "$out" ]]; then echo "已存在 $out，跳过下载"; continue; fi
  ok=0
  for host in "${HF_HOSTS[@]}"; do
    url="$host/$repo/resolve/main/$f"
    echo "下载 $url ..."
    if curl -fL --retry 3 -o "$out.part" "$url"; then mv "$out.part" "$out"; ok=1; break; fi
    rm -f "$out.part"
  done
  [[ $ok == 1 ]] || { echo "下载失败：$f（检查网络，或用 HF_HOST=... 换镜像）"; exit 1; }
done

# ---- 2. 编译补丁版 llama-funasr-cli（已存在则跳过；FORCE_BUILD=1 强制重编）----
BIN="$BIN_DIR/llama-funasr-cli"
if [[ -s "$BIN" && "${FORCE_BUILD:-0}" != "1" ]]; then
  echo "已存在 $BIN，跳过编译（FORCE_BUILD=1 可强制重编）"
else
  command -v cmake >/dev/null 2>&1 || {
    echo "未找到 cmake，尝试装到当前 python 环境 ..."
    python -m pip install -q cmake
  }
  command -v cmake >/dev/null 2>&1 || { echo "需要 cmake（brew install cmake 或 pip install cmake）"; exit 1; }
  SRC="$(dirname "$0")/../llamacpp/funasr-cli.cpp"
  WORK="$(mktemp -d /tmp/funasr-build.XXXXXX)"
  trap 'rm -rf "$WORK"' EXIT
  echo "拉取 Fun-ASR 源码（tag runtime-llamacpp-v0.2.1）到 $WORK ..."
  git clone -q --depth 1 --branch runtime-llamacpp-v0.2.1 \
    "${FUNASR_GIT_URL:-https://github.com/QwenAudio/Fun-ASR.git}" "$WORK"
  cp "$SRC" "$WORK/runtime/llama.cpp/funasr-cli/funasr-cli.cpp"
  echo "编译中（约 10-25 分钟，首次还会联网拉 llama.cpp 源码）..."
  cmake -S "$WORK/runtime/llama.cpp" -B "$WORK/runtime/llama.cpp/build" \
    -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$WORK/runtime/llama.cpp/build" --target llama-funasr-cli \
    -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" >/dev/null
  cp "$WORK/runtime/llama.cpp/build/bin/llama-funasr-cli" "$BIN"
  echo "编译完成：$BIN"
fi

echo
echo "全部就绪。config.yaml 无需填 llamacpp_bin/llamacpp_dir（默认即用 $DEST），"
echo "如需指定可填：llamacpp_bin: $BIN / llamacpp_dir: $GGUF_DIR / llamacpp_llm: $LLM"
