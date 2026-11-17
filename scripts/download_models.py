#!/usr/bin/env python3
"""
Optional: pre-download lyrics models into <project>/models/.

Normal usage does NOT require this script — align_lyrics.py will download
models to the system Hugging Face cache on first run.

Use this only if you want local copies under models/ (e.g. faster repeat runs
on the same machine without re-resolving cache paths).

Accuracy-first defaults (Python 3.10 + faster-whisper + transformers 5.x):
  - Systran/faster-whisper-medium  -> models/whisper-medium/
  - lemmonyjiang/opus-mt-ja-zh-jav -> models/opus-mt-ja-zh/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WHISPER_REPO = "Systran/faster-whisper-medium"
WHISPER_FOLDER = "whisper-medium"
MT_REPO = "lemmonyjiang/opus-mt-ja-zh-jav"
MT_FOLDER = "opus-mt-ja-zh"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def download_whisper(target: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"[1/2] 下载 Whisper 对齐模型: {WHISPER_REPO}")
    print(f"      -> {target}")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=WHISPER_REPO,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    if not (target / "model.bin").is_file():
        raise RuntimeError(f"Whisper 模型不完整，缺少 model.bin: {target}")
    print("      Whisper 下载完成")


def download_translation(target: Path) -> None:
    print(f"[2/2] 下载日译中模型: {MT_REPO}")
    print(f"      -> {target}")
    target.mkdir(parents=True, exist_ok=True)

    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(MT_REPO)
    model = MarianMTModel.from_pretrained(MT_REPO)
    tokenizer.save_pretrained(str(target))
    model.save_pretrained(str(target))

    if not (target / "config.json").is_file():
        raise RuntimeError(f"翻译模型不完整，缺少 config.json: {target}")
    print("      翻译模型下载完成")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="（可选）预下载歌词模型到项目 models/ 目录"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="模型根目录，默认 <项目根>/models",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models_dir = args.models_dir.resolve() if args.models_dir else project_root() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print(" VJ 歌词模型预下载（可选）")
    print(f" 目标目录: {models_dir}")
    print(" 需要联网，体积约 1.5~2 GB")
    print(" 不运行本脚本也可以：首次跑歌词功能时会自动下载到系统缓存")
    print("=" * 50)

    try:
        download_whisper(models_dir / WHISPER_FOLDER)
        download_translation(models_dir / MT_FOLDER)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print()
    print("全部完成。目录结构:")
    print(f"  {models_dir / WHISPER_FOLDER}")
    print(f"  {models_dir / MT_FOLDER}")
    print("align_lyrics.py 会优先使用上述本地目录（若存在）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
