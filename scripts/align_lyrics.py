#!/usr/bin/env python3
"""
Align provided Japanese lyrics to a vocal track, then export:
  - .lrc  : Chinese lines with timestamps (draft)
  - .txt  : Romaji / Japanese / Chinese per line (3 lines per sentence)

Models are loaded from Hugging Face on first run (cached under the user profile).
Optional local copies under <project>/models/ are used when present.

Usage:
  python scripts/align_lyrics.py ^
    --vocals path/to/vocals.flac ^
    --lyrics path/to/lyrics.txt ^
    --out-lrc path/to/output.lrc ^
    --out-txt path/to/output.txt
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI

DEFAULT_WHISPER_SIZE = "medium"
DEFAULT_HF_TOKEN_FILE = "hf_token.txt"
DEFAULT_DOUBAO_TOKEN_FILE = "doubao.txt"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
# 火山方舟推理接入点 ID（非裸模型名）；走接入点才可计入创作激励计划返 token。
DEFAULT_DOUBAO_ENDPOINT = ""
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_LLM_BACKEND = "doubao"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
OFFICIAL_HF_ENDPOINT = "https://huggingface.co"
LLM_BATCH_SIZE = 5
LLM_SINGLE_LINE_SPLIT_RETRIES = 2
LLM_BATCH_RETRIES = 1
_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
_KATAKANA_RE = re.compile(r"[\u30a0-\u30ff]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# Helsinki-NLP/opus-mt-ja-zh 已从 HF 下架。
# shun89/opus-mt-ja-zh 仅有 pytorch_model.bin，transformers 5.x + torch<2.6 无法加载。
# 改用带 safetensors 的日译中 Marian 模型（基于 shun89 微调）。
# TRANSLATION_REPO = "lemmonyjiang/opus-mt-ja-zh-jav"
# TRANSLATION_DIR_NAME = "opus-mt-ja-zh"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_ffmpeg_exe() -> Path:
    return project_root() / "ffmpeg" / "bin" / "ffmpeg.exe"


def configure_ffmpeg() -> Path:
    """Put bundled ffmpeg on PATH for stable_whisper audio decoding."""
    ffmpeg_exe = resolve_ffmpeg_exe()
    if not ffmpeg_exe.is_file():
        raise FileNotFoundError(
            f"未找到内置 FFmpeg: {ffmpeg_exe}\n"
            "请确认项目根目录存在 ffmpeg\\bin\\ffmpeg.exe。"
        )

    ffmpeg_bin = str(ffmpeg_exe.parent)
    path = os.environ.get("Path", "")
    if ffmpeg_bin not in path.split(os.pathsep):
        os.environ["Path"] = ffmpeg_bin + os.pathsep + path
    os.environ["FFMPEG_BINARY"] = str(ffmpeg_exe)
    return ffmpeg_exe


def resolve_models_dir(explicit: Path | None) -> Path:
    return explicit.resolve() if explicit else project_root() / "models"


def configure_hf_token(*, cli_token: str | None = None, token_file: Path | None = None) -> None:
    """Configure HF_TOKEN before any huggingface_hub download.

    Priority: --hf-token > environment variable > hf_token.txt in project root.
    """
    if cli_token and cli_token.strip():
        os.environ["HF_TOKEN"] = cli_token.strip()
        print("[info] 已使用命令行 HF Token")
        return

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("[info] 已使用环境变量 HF Token")
        return

    path = token_file.resolve() if token_file else project_root() / DEFAULT_HF_TOKEN_FILE
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            print(f"[info] 已使用本地 Token 文件: {path}")
            return

    print(
        "[info] 未配置 HF Token，将匿名下载（可能较慢）。"
        f"可在项目根目录创建 {DEFAULT_HF_TOKEN_FILE} 存放 Read Token。",
        file=sys.stderr,
    )

def configure_doubao_token(*, cli_token: str | None = None) -> None:
    """Configure ARK_API_KEY for Volcengine Ark (Doubao).

    Priority: --ark-api-key > ARK_API_KEY > DOUBAO_TOKEN > doubao.txt in project root.
    """
    if cli_token and cli_token.strip():
        os.environ["ARK_API_KEY"] = cli_token.strip()
        print("[info] 已使用命令行 Ark API Key")
        return

    if os.environ.get("ARK_API_KEY"):
        print("[info] 已使用环境变量 ARK_API_KEY")
        return

    if os.environ.get("DOUBAO_TOKEN"):
        os.environ["ARK_API_KEY"] = os.environ["DOUBAO_TOKEN"]
        print("[info] 已使用环境变量 DOUBAO_TOKEN")
        return

    path = project_root() / DEFAULT_DOUBAO_TOKEN_FILE
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            os.environ["ARK_API_KEY"] = token
            print(f"[info] 已使用本地 Token 文件: {path}")
            return

    raise RuntimeError(
        "未配置豆包 API Key。请设置 ARK_API_KEY / DOUBAO_TOKEN，"
        f"或在项目根目录创建 {DEFAULT_DOUBAO_TOKEN_FILE}。"
    )
    
def configure_hf_endpoint() -> None:
    """Use HF mirror in China when HF_ENDPOINT is not already set."""
    if os.environ.get("HF_ENDPOINT"):
        print(f"[info] 已使用环境变量 HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
    else:
        os.environ["HF_ENDPOINT"] = DEFAULT_HF_ENDPOINT
        print(f"[info] 使用 Hugging Face 镜像: {DEFAULT_HF_ENDPOINT}")
    # 部分 huggingface_hub 版本读取此变量
    if not os.environ.get("HUGGINGFACE_HUB_ENDPOINT"):
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = os.environ["HF_ENDPOINT"]


def configure_hf_cache() -> Path:
    """Use project-local HF cache so packaged DS has a stable, portable cache dir."""
    cache_root = project_root() / "models" / "hf-cache"
    hub_cache = cache_root / "hub"
    cache_root.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return hub_cache


def sync_hf_token_env() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def _apply_hf_endpoint(endpoint: str) -> None:
    endpoint = endpoint.rstrip("/")
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = endpoint


def _whisper_download_endpoints() -> list[str]:
    endpoints: list[str] = []
    for candidate in (
        os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT),
        DEFAULT_HF_ENDPOINT,
        OFFICIAL_HF_ENDPOINT,
    ):
        normalized = (candidate or "").rstrip("/")
        if normalized and normalized not in endpoints:
            endpoints.append(normalized)
    return endpoints


def ensure_whisper_model(models_dir: Path, whisper_size: str) -> str:
    """Ensure CT2 Whisper model exists under models/whisper-{size}/."""
    local_dir = models_dir / f"whisper-{whisper_size}"
    if (local_dir / "model.bin").is_file():
        print(f"[info] 使用本地 Whisper 模型: {local_dir}")
        return str(local_dir)

    from faster_whisper.utils import download_model

    hub_cache = configure_hf_cache()
    print(f"[info] Hugging Face 缓存: {hub_cache}")

    models_dir.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for endpoint in _whisper_download_endpoints():
        _apply_hf_endpoint(endpoint)
        print(
            f"[info] 正在从 {endpoint} 下载 Whisper {whisper_size} "
            f"到 {local_dir} ..."
        )
        try:
            if local_dir.exists():
                for child in local_dir.iterdir():
                    if child.is_file():
                        child.unlink()
            downloaded = download_model(
                whisper_size,
                output_dir=str(local_dir),
                local_files_only=False,
            )
            model_dir = local_dir if (local_dir / "model.bin").is_file() else Path(downloaded)
            if (model_dir / "model.bin").is_file():
                print(f"[info] Whisper 模型下载完成: {model_dir}")
                return str(model_dir)
            raise RuntimeError(f"下载完成但未找到 model.bin: {model_dir}")
        except Exception as exc:
            last_error = exc
            print(f"[warn] 从 {endpoint} 下载失败: {exc}", file=sys.stderr)

    raise RuntimeError(
        f"Whisper {whisper_size} 模型下载失败。"
        "hf-mirror 与当前 huggingface_hub 可能不兼容，程序已自动尝试官方源。"
        "请确认可访问 https://huggingface.co 或手动将模型放入 "
        f"{local_dir} 。"
        f"最后错误: {last_error}"
    ) from last_error


# def resolve_translation_model(models_dir: Path) -> str:
#     local_dir = models_dir / TRANSLATION_DIR_NAME
#     if (local_dir / "config.json").is_file():
#         print(f"[info] 使用本地翻译模型: {local_dir}")
#         return str(local_dir)

#     print(
#         f"[info] 将加载翻译模型: {TRANSLATION_REPO}"
#         "（首次使用需联网下载，之后使用系统缓存）"
#     )
#     return TRANSLATION_REPO


def is_local_dir(model_ref: str) -> bool:
    path = Path(model_ref)
    return path.is_dir()


# ---------------------------------------------------------------------------
# Lyrics I/O
# ---------------------------------------------------------------------------


def read_lyric_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"歌词文件为空: {path}")
    return lines


def format_lrc_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"[{minutes:02d}:{secs:05.2f}]"


# ---------------------------------------------------------------------------
# Romaji + translation (in-memory)
# ---------------------------------------------------------------------------


# def build_romaji_converter():
#     from pykakasi import kakasi

#     converter = kakasi()

#     def to_romaji(japanese: str) -> str:
#         parts = converter.convert(japanese)
#         tokens = [item.get("hepburn", "").strip() for item in parts]
#         tokens = [token for token in tokens if token]
#         return " ".join(tokens) if tokens else japanese

#     return to_romaji


# def build_translator(model_ref: str):
    # from transformers import MarianMTModel, MarianTokenizer

    # local_only = is_local_dir(model_ref)
    # print("[align] 正在下载/加载翻译模型，请稍候 ...", flush=True)
    # tokenizer = MarianTokenizer.from_pretrained(model_ref, local_files_only=local_only)
    # print("[align] 翻译 tokenizer 就绪，正在加载权重 ...", flush=True)
    # model = MarianMTModel.from_pretrained(model_ref, local_files_only=local_only)
    # print("[align] 翻译模型加载完成", flush=True)

    # def translate(japanese: str) -> str:
    #     batch = tokenizer([japanese], return_tensors="pt", padding=True)
    #     output = model.generate(**batch)
    #     return tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # return translate

def _resolve_llm_dtype():
    """CPU 推理用半精度即可；歌词任务不必 float32，3B 可少占约一半内存。"""
    import torch

    return torch.float16


def load_llm(model_id:str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_only = is_local_dir(model_id)
    llm_dtype = _resolve_llm_dtype()
    print(f"[llm] 加载模型: {model_id} ({llm_dtype})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=llm_dtype,
        trust_remote_code=True,
        local_files_only=local_only,
    )

    model.eval()
    print("[llm] 模型就绪", flush=True)

    return model,tokenizer

def _clean_llm_response(response: str) -> str:
    response = response.strip()
    if response.startswith("```"):
        response = re.sub(r"^```(?:json)?\s*", "", response, flags=re.IGNORECASE)
        response = re.sub(r"\s*```$", "", response).strip()
    start = response.find("[")
    end = response.rfind("]")
    if start >= 0 and end > start:
        return response[start : end + 1]
    return response


def _merge_llm_items(items: list[dict]) -> dict:
    romaji_parts: list[str] = []
    chinese_parts: list[str] = []
    for item in items:
        romaji = str(item.get("romaji", "")).strip()
        chinese = str(item.get("chinese", "")).strip()
        if romaji:
            romaji_parts.append(romaji)
        if chinese:
            chinese_parts.append(chinese)
    return {
        "romaji": " ".join(romaji_parts),
        "chinese": " ".join(chinese_parts),
    }


class LLMCountMismatchError(RuntimeError):
    def __init__(self, actual: int, expected: int, data: list[dict], response: str):
        self.actual = actual
        self.expected = expected
        self.data = data
        self.response = response
        super().__init__(f"LLM 返回 {actual} 条，期望 {expected} 条")


def _coalesce_llm_items(data: list[dict]) -> list[dict]:
    """Merge alternating {\"romaji\":...} / {\"chinese\":...} objects into pairs."""
    recovered: list[dict] = []
    pending_romaji = ""

    for item in data:
        romaji = str(item.get("romaji", "")).strip()
        chinese = str(item.get("chinese", "")).strip()

        if romaji and chinese:
            recovered.append({"romaji": romaji, "chinese": chinese})
            pending_romaji = ""
        elif romaji:
            if pending_romaji:
                recovered.append({"romaji": pending_romaji, "chinese": ""})
            pending_romaji = romaji
        elif chinese:
            if pending_romaji:
                recovered.append({"romaji": pending_romaji, "chinese": chinese})
                pending_romaji = ""
            else:
                recovered.append({"romaji": "", "chinese": chinese})

    if pending_romaji:
        recovered.append({"romaji": pending_romaji, "chinese": ""})
    return recovered


def _recover_llm_array(data: list[dict], expected: int) -> list[dict] | None:
    if len(data) == expected and all(
        str(item.get("romaji", "")).strip() and str(item.get("chinese", "")).strip()
        for item in data
    ):
        return data

    coalesced = _coalesce_llm_items(data)
    if len(coalesced) == expected:
        return coalesced
    return None


def _save_llm_debug_response(response: str) -> Path:
    debug_path = Path(tempfile.gettempdir()) / "vj_llm_last_response.txt"
    debug_path.write_text(response, encoding="utf-8")
    return debug_path


def _sanitize_llm_json_text(text: str) -> str:
    text = re.sub(r"\}\}\s*\]", "}]", text)
    text = re.sub(r"\}\}\s*,", "},", text)
    text = re.sub(r",\s*\]", "]", text)
    return text


_LLM_OBJECT_RE = re.compile(
    r'\{\s*"romaji"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"chinese"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
    re.DOTALL,
)


def _extract_llm_objects_regex(text: str) -> list[dict]:
    items: list[dict] = []
    for match in _LLM_OBJECT_RE.finditer(text):
        items.append(
            {
                "romaji": match.group(1).replace('\\"', '"'),
                "chinese": match.group(2).replace('\\"', '"'),
            }
        )
    return items


def _parse_llm_json_array_strict(response: str, expected: int) -> list[dict]:
    import json

    cleaned = _clean_llm_response(response)
    candidates: list[str] = []
    for variant in (cleaned, _sanitize_llm_json_text(cleaned)):
        if variant and variant not in candidates:
            candidates.append(variant)
        escaped = re.sub(r"\\u(?![0-9a-fA-F]{4})", r"\\\\u", variant)
        if escaped not in candidates:
            candidates.append(escaped)

    last_err: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_err = exc
            continue
        if not isinstance(data, list):
            raise RuntimeError("LLM 返回的不是 JSON 数组")
        normalized = _finalize_llm_array(data, expected, response=response)
        if normalized is not None:
            return normalized

    regex_items = _extract_llm_objects_regex(cleaned)
    if regex_items:
        normalized = _finalize_llm_array(regex_items, expected, response=response)
        if normalized is not None:
            print(
                f"[warn] JSON 语法损坏，已从文本中提取 {len(normalized)} 条",
                file=sys.stderr,
            )
            return normalized

    debug_path = _save_llm_debug_response(response)
    raise RuntimeError(
        f"无法解析 LLM 返回的 JSON: {last_err}。"
        f"完整回复已保存: {debug_path}"
    )


def _finalize_llm_array(
    data: list[dict],
    expected: int,
    *,
    response: str,
) -> list[dict] | None:
    if len(data) == expected and all(
        str(item.get("romaji", "")).strip() and str(item.get("chinese", "")).strip()
        for item in data
    ):
        return data

    recovered = _recover_llm_array(data, expected)
    if recovered is not None:
        if len(data) != expected:
            print(
                f"[warn] LLM 返回 {len(data)} 条，已修复为 {expected} 条",
                file=sys.stderr,
            )
        else:
            print(
                f"[warn] LLM JSON 字段拆散，已修复为 {expected} 条",
                file=sys.stderr,
            )
        return recovered

    if len(data) != expected:
        raise LLMCountMismatchError(len(data), expected, data, response)
    return None


def _latin_letters_in_ascii_words(text: str) -> int:
    return sum(len(word) for word in re.findall(r"[A-Za-z]+", text))


def _romaji_latin_letters(romaji: str) -> int:
    return sum(1 for ch in romaji if ch.isascii() and ch.isalpha())


def _split_response_likely_incomplete(items: list[dict], japanese: str) -> bool:
    """Heuristic: model split one line but a fragment missed Japanese before parentheses."""
    if len(items) < 2:
        return False

    before_paren = re.split(r"[（(]", japanese, maxsplit=1)[0]
    kana_before_paren = _HIRAGANA_RE.findall(before_paren) + _KATAKANA_RE.findall(before_paren)
    if len(kana_before_paren) < 3:
        return False

    first_romaji = str(items[0].get("romaji", "")).strip()
    if not first_romaji:
        return True

    english_budget = _latin_letters_in_ascii_words(before_paren)
    if _romaji_latin_letters(first_romaji) <= english_budget + 2:
        return True
    return False


def build_lyrics_prompt(
    lines: list[str],
    *,
    retry: bool = False,
    split_retry: bool = False,
    format_retry: bool = False,
    romaji_retry: bool = False,
) -> str:
    numbered = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
    retry_note = (
        "上次输出不合格：chinese 必须是简体中文（禁止抄日文），"
        "romaji 必须是拉丁字母 Hepburn 罗马音（日文部分不能仍是假名/汉字）。请重新输出。\n"
        if retry
        else ""
    )
    split_retry_note = (
        "上次你把 1 句歌词拆成了多条 JSON，且漏译了句中日文。"
        "这次必须只输出 1 个 JSON 对象，整句（含括号）一起翻译，禁止拆分。\n"
        if split_retry
        else ""
    )
    romaji_retry_note = (
        "上次 romaji 里混入了日文汉字/假名。romaji 必须全部用拉丁字母，"
        "例如 超えて→koete、燃やせ→moyase、瞬間で→shunkan de。\n"
        if romaji_retry
        else ""
    )
    format_retry_note = (
        "上次 JSON 格式错误：不要把 romaji 和 chinese 拆成两个对象。"
        "每个元素必须同时包含 romaji 和 chinese 两个字段。\n"
        if format_retry
        else ""
    )
    single_line_note = (
        "当前只有 1 句歌词（括号内容也算同一句）。"
        "JSON 数组必须恰好 1 个元素，禁止拆成多条。\n"
        if len(lines) == 1
        else ""
    )
    return f"""你是 J-Pop 歌词翻译助手。对下面每句日文歌词，输出：
1) romaji：Hepburn 罗马音，适合跟唱，保持准确性；英文部分保持英文
2) chinese：自然简体中文，翻译整句含义（含括号内容）；英文口号也要译成中文
{retry_note}{split_retry_note}{romaji_retry_note}{format_retry_note}{single_line_note}
硬性要求：
【chinese 风格】
- 用简体中文，不要假名，不要复制日文原文
- 跟唱歌词体，主句宜短；含括号的整句可更长，括号内也必须完整翻译，不要省略
- 片假名外来语/惯用语意译，禁止音译（例：マンネリ→循规蹈矩/千篇一律，ありがち→老套/常见）
- 抽象、意象、生理描写句按歌意意译，禁止解剖式直译（例：脳幹降りて溶けてく→意识沉下去逐渐融化，不要写「脑干」）
- 全曲重复句（副歌）译法必须完全一致

【romaji 风格】
- 用拉丁字母，不要输出汉字/假名
- Hepburn，英文原样保留
- 复合动词/促音连写：取り払って→toriharatte，超えて→koete，踏み込んで→fumikonde
- 片假名按实际发音：マンネリ→manneri

【格式与对应】
- 输入有几句，JSON 数组就有几个元素，顺序一一对应，不要合并或拆分
- 每个数组元素必须同时有 romaji 和 chinese，禁止分成两个对象

必须只输出 JSON 数组，长度必须是 {len(lines)}，格式：
[{{"romaji":"...","chinese":"..."}}, {{"romaji":"...","chinese":"..."}}]
错误示例（禁止）：[{{"romaji":"..."}}, {{"chinese":"..."}}]
错误示例（禁止）：{{"romaji":"4Step 超えて Floor 燃やせ"}}(romaji 不能含日文）
错误示例（禁止）：[{{"romaji":"a","chinese":"b"}}}}]（括号多写、非法 JSON)
正确示例：{{"romaji":"manneri de itsu no ma ni ka","chinese":"循规蹈矩间 不知不觉"}}
正确示例：{{"romaji":"4 Step koete Floor moyase (Don't stop! Don't stop!)","chinese":"跨过四步 点燃舞池（别停！别停！一步接一步向前走！）"}}
不要输出 markdown,不要解释,只输出一行合法 JSON 数组。
歌词：
{numbered}
"""


def _estimate_max_new_tokens(line_count: int) -> int:
    return min(4096, max(1024, line_count * 160))


def _romaji_looks_invalid(romaji: str, japanese: str) -> bool:
    romaji = romaji.strip()
    japanese = japanese.strip()
    if not romaji:
        return True
    if romaji == japanese:
        return True

    latin_count = sum(1 for ch in romaji if ch.isascii() and ch.isalpha())
    cjk_count = len(_CJK_RE.findall(romaji))
    kana_count = len(_HIRAGANA_RE.findall(romaji)) + len(_KATAKANA_RE.findall(romaji))
    jp_kana_count = len(_HIRAGANA_RE.findall(japanese)) + len(_KATAKANA_RE.findall(japanese))

    if cjk_count > 0 or kana_count >= 2:
        return True
    if jp_kana_count >= 2 and latin_count < 2:
        return True
    return False


def _chinese_looks_invalid(chinese: str, japanese: str) -> bool:
    chinese = chinese.strip()
    japanese = japanese.strip()
    if not chinese:
        return True

    norm_cn = normalize_lyric_text(chinese)
    norm_jp = normalize_lyric_text(japanese)
    if norm_cn == norm_jp:
        return True
    if norm_jp and norm_jp in norm_cn and len(chinese) <= len(japanese) + 8:
        return True

    cjk_count = len(_CJK_RE.findall(chinese))
    kana_count = len(_HIRAGANA_RE.findall(chinese)) + len(_KATAKANA_RE.findall(chinese))
    jp_kana_count = len(_HIRAGANA_RE.findall(japanese)) + len(_KATAKANA_RE.findall(japanese))

    if kana_count >= 2 and cjk_count < 2:
        return True
    if jp_kana_count >= 2 and cjk_count == 0:
        return True
    return False


def _line_result_looks_invalid(romaji: str, chinese: str, japanese: str) -> bool:
    return _romaji_looks_invalid(romaji, japanese) or _chinese_looks_invalid(chinese, japanese)


def _chinese_is_acceptable(chinese: str, japanese: str) -> bool:
    return not _chinese_looks_invalid(chinese, japanese)


def _llm_generate_response(
    model,
    tokenizer,
    lines: list[str],
    *,
    retry: bool = False,
    split_retry: bool = False,
    format_retry: bool = False,
    romaji_retry: bool = False,
) -> str:
    import torch

    prompt = build_lyrics_prompt(
        lines,
        retry=retry,
        split_retry=split_retry,
        format_retry=format_retry,
        romaji_retry=romaji_retry,
    )
    message = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=_estimate_max_new_tokens(len(lines)),
            do_sample=False,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _pairs_from_llm_data(data: list[dict]) -> list[tuple[str, str]]:
    return [(str(item["romaji"]).strip(), str(item["chinese"]).strip()) for item in data]


def _try_merge_split_response(
    items: list[dict],
    japanese: str,
) -> tuple[str, str] | None:
    if len(items) < 2:
        return None
    if _split_response_likely_incomplete(items, japanese):
        return None

    merged = _merge_llm_items(items)
    romaji = merged["romaji"]
    chinese = merged["chinese"]
    if _line_result_looks_invalid(romaji, chinese, japanese):
        return None
    return romaji, chinese


def _invoke_llm_single_line(
    model,
    tokenizer,
    line: str,
    *,
    retry: bool = False,
) -> tuple[str, str]:
    last_mismatch: LLMCountMismatchError | None = None
    last_candidate: tuple[str, str] | None = None
    last_response = ""

    for attempt in range(1 + LLM_SINGLE_LINE_SPLIT_RETRIES):
        split_retry = attempt > 0 or last_mismatch is not None
        romaji_retry = (
            last_candidate is not None
            and _chinese_is_acceptable(last_candidate[1], line)
            and _romaji_looks_invalid(last_candidate[0], line)
        )
        if split_retry:
            print(
                f"[llm] 单句被拆成多条，重试 ({attempt + 1}/{1 + LLM_SINGLE_LINE_SPLIT_RETRIES}) ...",
                flush=True,
            )
        elif romaji_retry:
            print(
                f"[llm] 单句 romaji 含日文，重试 ({attempt + 1}/{1 + LLM_SINGLE_LINE_SPLIT_RETRIES}) ...",
                flush=True,
            )

        response = _llm_generate_response(
            model,
            tokenizer,
            [line],
            retry=retry or split_retry or romaji_retry,
            split_retry=split_retry,
            romaji_retry=romaji_retry,
        )
        last_response = response
        try:
            data = _parse_llm_json_array_strict(response, 1)
            romaji, chinese = _pairs_from_llm_data(data)[0]
            last_candidate = (romaji, chinese)
            if not _line_result_looks_invalid(romaji, chinese, line):
                return romaji, chinese
            if _romaji_looks_invalid(romaji, line) and _chinese_is_acceptable(chinese, line):
                print("[llm] 中文可用但 romaji 不合格，继续重试 ...", flush=True)
            else:
                print("[llm] 单句结果不合格，继续重试 ...", flush=True)
        except LLMCountMismatchError as exc:
            last_mismatch = exc
            print(
                f"[warn] 单句返回 {exc.actual} 条 JSON，期望 1 条",
                file=sys.stderr,
            )

    if last_mismatch and last_mismatch.actual > 1:
        merged_pair = _try_merge_split_response(last_mismatch.data, line)
        if merged_pair is not None:
            print(
                "[warn] 多次重试仍被拆分，使用通过校验的合并结果",
                file=sys.stderr,
            )
            return merged_pair
        print(
            "[warn] 拆分响应疑似漏译，合并结果未采用",
            file=sys.stderr,
        )

    if last_candidate and _chinese_is_acceptable(last_candidate[1], line):
        print(
            "[warn] romaji 仍不合格，中文可用，降级采用: "
            f"romaji={last_candidate[0]} | chinese={last_candidate[1]}",
            file=sys.stderr,
        )
        return last_candidate

    if last_mismatch:
        debug_path = _save_llm_debug_response(last_mismatch.response)
        raise RuntimeError(
            f"单句歌词无法得到合格翻译（返回 {last_mismatch.actual} 条）。"
            f"完整回复已保存: {debug_path}"
        )

    debug_path = _save_llm_debug_response(last_response)
    raise RuntimeError(f"单句歌词翻译不合格。完整回复已保存: {debug_path}")


def _invoke_llm_batch(model, tokenizer, lines: list[str], *, retry: bool = False) -> list[tuple[str, str]]:
    last_error: Exception | None = None

    for attempt in range(1 + LLM_BATCH_RETRIES):
        format_retry = attempt > 0
        if format_retry:
            print(
                f"[llm] 批次 JSON 格式异常，重试 ({attempt + 1}/{1 + LLM_BATCH_RETRIES}) ...",
                flush=True,
            )
        response = _llm_generate_response(
            model,
            tokenizer,
            lines,
            retry=retry or format_retry,
            format_retry=format_retry,
        )
        try:
            data = _parse_llm_json_array_strict(response, len(lines))
            return _pairs_from_llm_data(data)
        except (LLMCountMismatchError, RuntimeError) as exc:
            last_error = exc
            print(f"[warn] 批次生成失败: {exc}", file=sys.stderr)

    print(
        f"[warn] 批次 {len(lines)} 句改为逐句生成 ...",
        file=sys.stderr,
    )
    pairs: list[tuple[str, str]] = []
    for line in lines:
        pairs.append(_invoke_llm_single_line(model, tokenizer, line, retry=True))
    if last_error is not None:
        print(f"[info] 批次失败原因: {last_error}", flush=True)
    return pairs


def _invoke_llm(model, tokenizer, lines: list[str], *, retry: bool = False) -> list[tuple[str, str]]:
    if len(lines) == 1:
        return [_invoke_llm_single_line(model, tokenizer, lines[0], retry=retry)]
    return _invoke_llm_batch(model, tokenizer, lines, retry=retry)


def _iter_llm_batches(total: int) -> list[tuple[int, int]]:
    batches: list[tuple[int, int]] = []
    for batch_start in range(0, total, LLM_BATCH_SIZE):
        batches.append((batch_start, min(batch_start + LLM_BATCH_SIZE, total)))
    return batches


def _retry_invalid_lines(
    model,
    tokenizer,
    lines: list[str],
    pairs: list[tuple[str, str] | None],
    indices: list[int],
    *,
    label: str,
) -> None:
    invalid_indices = [
        index
        for index in indices
        if pairs[index] is None
        or _line_result_looks_invalid(pairs[index][0], pairs[index][1], lines[index])
    ]
    if not invalid_indices:
        return

    print(f"[llm] {label}：{len(invalid_indices)} 句不合格，逐句重试 ...", flush=True)
    for index in invalid_indices:
        line = lines[index]
        print(f"[llm] 重试第 {index + 1} 句: {line}", flush=True)
        romaji, chinese = _invoke_llm(model, tokenizer, [line], retry=True)[0]
        if _line_result_looks_invalid(romaji, chinese, line):
            print(
                f"[warn] 第 {index + 1} 句仍不理想: romaji={romaji} | chinese={chinese}",
                file=sys.stderr,
            )
        pairs[index] = (romaji, chinese)


def _run_llm_batches(
    model,
    tokenizer,
    lines: list[str],
    pairs: list[tuple[str, str] | None],
    batch_plan: list[tuple[int, int]],
    total: int,
) -> None:
    for batch_index, (batch_start, batch_end) in enumerate(batch_plan, start=1):
        batch = lines[batch_start:batch_end]
        label = (
            f"生成第 {batch_index}/{len(batch_plan)} 批 "
            f"({len(batch)} 句，{batch_start + 1}-{batch_end}/{total})"
        )
        print(f"[llm] {label} ...", flush=True)
        batch_pairs = _invoke_llm(model, tokenizer, batch)
        for offset, pair in enumerate(batch_pairs):
            pairs[batch_start + offset] = pair

        batch_indices = list(range(batch_start, batch_end))
        _retry_invalid_lines(
            model,
            tokenizer,
            lines,
            pairs,
            batch_indices,
            label=f"第 {batch_index} 批校验",
        )


def generate_romaji_and_chinese(model, tokenizer, lines: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str] | None] = [None] * len(lines)
    total = len(lines)
    batch_plan = _iter_llm_batches(total)
    _run_llm_batches(model, tokenizer, lines, pairs, batch_plan, total)

    _retry_invalid_lines(
        model,
        tokenizer,
        lines,
        pairs,
        list(range(total)),
        label="最终校验",
    )

    return [pair for pair in pairs if pair is not None]


def create_doubao_client() -> OpenAI:
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("ARK_API_KEY 未配置，请先调用 configure_doubao_token()")
    return OpenAI(base_url=DEFAULT_ARK_BASE_URL, api_key=api_key)


def extract_doubao_response_text(response) -> str:
    """Extract assistant text from Ark responses.create (skips reasoning blocks)."""
    chunks: list[str] = []

    output = getattr(response, "output", None) or []
    for item in output:
        item_type = getattr(item, "type", None)
        if item_type != "message":
            continue
        for block in getattr(item, "content", None) or []:
            block_type = getattr(block, "type", None)
            if block_type in ("output_text", "text"):
                text = getattr(block, "text", None)
                if text:
                    chunks.append(str(text).strip())

    if chunks:
        return "\n".join(chunks)

    # Fallback for chat.completions-style compatibility
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message else None
        if content:
            return str(content).strip()

    raise RuntimeError("豆包响应中未找到 assistant 文本内容")


def generate_romaji_and_chinese_doubao(
    lines: list[str],
    *,
    endpoint_id: str = DEFAULT_DOUBAO_ENDPOINT,
) -> list[tuple[str, str]]:
    client = create_doubao_client()
    prompt = build_lyrics_prompt(lines)
    print(f"[llm] 豆包一次性生成 {len(lines)} 句 (接入点 {endpoint_id}) ...", flush=True)

    last_error: Exception | None = None
    for attempt in range(2):
        if attempt > 0:
            print("[llm] 豆包 JSON 解析失败，重试 ...", flush=True)

        response = client.responses.create(
            model=endpoint_id,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )
        text = extract_doubao_response_text(response)
        try:
            data = _parse_llm_json_array_strict(text, len(lines))
            return _pairs_from_llm_data(data)
        except RuntimeError as exc:
            last_error = exc
            _save_llm_debug_response(text)

    raise RuntimeError(f"豆包返回无法解析为 {len(lines)} 条 JSON: {last_error}")


_PUNCT_RE = re.compile(
    r"[\s\(\)（）\[\]【】!！?？,，.．·:：;；\"'＂＇\-—…]+"
)

# Whisper 常输出平假名，歌词文件可能是片假名，统一后再比较。
_KATAKANA_START = ord("ァ")
_KATAKANA_END = ord("ヶ")


def _katakana_to_hiragana(text: str) -> str:
    chars: list[str] = []
    for ch in text:
        code = ord(ch)
        if _KATAKANA_START <= code <= _KATAKANA_END:
            chars.append(chr(code - 0x60))
        else:
            chars.append(ch)
    return "".join(chars)


def normalize_lyric_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT_RE.sub("", text)
    text = _katakana_to_hiragana(text)
    return text.casefold()


def _lyric_text_matches(needle: str, hay: str, *, allow_prefix: bool = False) -> bool:
    if not hay:
        return False
    if needle in hay or hay in needle:
        return True
    if not allow_prefix or len(needle) < 3:
        return False
    min_prefix = max(3, len(needle) * 2 // 3)
    for size in range(len(needle), min_prefix - 1, -1):
        if needle[:size] in hay:
            return True
    return False


def _log_lyric_hit(start: float, label: str, snippet: str) -> None:
    print(
        f"[align] 命中转写片段 [{start:.2f}s] ({label}): {snippet.strip()}",
        flush=True,
    )


def detect_lyrics_start_sec(model, vocals_path: Path, first_line: str) -> float:
    needle = normalize_lyric_text(first_line)
    if not needle:
        return 0.0

    print("[align] 正在转写人声以定位第一句开唱时间 ...", flush=True)

    result = model.transcribe(str(vocals_path), language="ja", verbose=False)

    if result is None or not result.segments:
        print("[warn] 转写无结果, intro 偏移置 0", file=sys.stderr)
        return 0.0

    segments = result.segments
    normalized = [normalize_lyric_text(segment.text or "") for segment in segments]

    for index, segment in enumerate(segments):
        hay = normalized[index]
        if _lyric_text_matches(needle, hay):
            start = float(segment.start or 0.0)
            _log_lyric_hit(start, "完整匹配", segment.text or "")
            return start

    max_window = 4
    for start_index in range(len(segments)):
        combined = ""
        snippet_parts: list[str] = []
        for offset in range(max_window):
            join_index = start_index + offset
            if join_index >= len(segments):
                break
            combined += normalized[join_index]
            snippet_parts.append((segments[join_index].text or "").strip())
            if _lyric_text_matches(needle, combined):
                start = float(segments[start_index].start or 0.0)
                _log_lyric_hit(start, "跨片段匹配", " / ".join(snippet_parts))
                return start

    for index, segment in enumerate(segments):
        hay = normalized[index]
        if _lyric_text_matches(needle, hay, allow_prefix=True):
            start = float(segment.start or 0.0)
            _log_lyric_hit(start, "前缀匹配", segment.text or "")
            return start

    print(
        f"[warn] 未在转写中找到第一句歌词，intro 偏移置 0。first_line={first_line.strip()}",
        file=sys.stderr,
    )
    return 0.0

def trim_audio_from(vocals_path:Path,start_sec:float) -> tuple[Path,float]:
    if start_sec <= 0.0:
        return vocals_path,0.0

    ffmpeg = resolve_ffmpeg_exe()
    trimmed = Path(tempfile.gettempdir()) / f"vj_align_trim_{vocals_path.stem}.flac"

    cmd = [str(ffmpeg), "-y","-ss",f"{start_sec:.3f}","-i",str(vocals_path),"-c","copy",str(trimmed)]
    
    print(f"[align] 从 {start_sec:.2f}s 裁切人声 -> {trimmed}", flush=True)
    subprocess.run(cmd,check=True)
    return trimmed,start_sec

# ---------------------------------------------------------------------------
# Forced alignment
# ---------------------------------------------------------------------------


@dataclass
class AlignedLine:
    start: float
    japanese: str
    romaji: str
    chinese: str


def align_lyrics_to_audio(
    vocals_path: Path,
    lyric_lines: list[str],
    *,
    whisper_model_ref: str,
    llm_backend: str,
    llm_model_id: str,
    doubao_endpoint_id: str,
    device: str,
    compute_type: str,
) -> list[AlignedLine]:
    import stable_whisper

    print(f"[align] 加载 Whisper 模型: {whisper_model_ref} ({device})")
    try:
        model = stable_whisper.load_faster_whisper(
            whisper_model_ref,
            device=device,
            compute_type=compute_type,
            local_files_only=is_local_dir(whisper_model_ref),
        )
    except Exception as exc:
        msg = str(exc)
        if any(
            key in msg.lower()
            for key in ("hub", "cache", "connection", "internet", "requested files")
        ):
            raise RuntimeError(
                "Whisper 模型下载/读取失败。请确认可联网访问 Hugging Face 镜像，"
                f"在项目根目录创建 {DEFAULT_HF_TOKEN_FILE} 存放 Read Token 后重试；"
                f"若仍失败，可删除 models/hf-cache 后重新下载，"
                f"或将 CT2 模型放入 models/whisper-{whisper_model_ref}/ 。"
                f"原始错误: {msg}"
            ) from exc
        raise

    # --- 新增：自动 intro ---
    lyrics_start = detect_lyrics_start_sec(model, vocals_path, lyric_lines[0])
    align_audio_path, time_offset = trim_audio_from(vocals_path, lyrics_start)
    if time_offset > 0:
        print(f"[align] 使用裁切后人声对齐，全局偏移 +{time_offset:.2f}s")
    else:
        print("[align] 使用完整人声对齐")    

    lyrics_text = "\n".join(lyric_lines)
    print(f"[align] 正在对齐 {len(lyric_lines)} 句歌词 ...")
    result = model.align(
        str(align_audio_path),
        lyrics_text,
        "ja",
        original_split=True,
        fast_mode=False,
        verbose=False,
    )

    if result is None or not result.segments:
        raise RuntimeError("对齐失败：未得到任何时间轴片段")

    segments = result.segments
    if len(segments) != len(lyric_lines):
        print(
            f"[warn] 歌词行数({len(lyric_lines)})与对齐片段数({len(segments)})不一致，"
            "将按较短长度配对",
            file=sys.stderr,
        )

    pair_count = min(len(segments), len(lyric_lines))

    # 对齐结束后释放 Whisper，避免与 LLM 同时占满 16GB 内存导致换页极慢
    del model
    import gc
    gc.collect()
    print("[align] Whisper 已释放，开始 LLM 生成 ...", flush=True)

    lyric_batch = lyric_lines[:pair_count]
    if llm_backend == "doubao":
        pairs = generate_romaji_and_chinese_doubao(
            lyric_batch, endpoint_id=doubao_endpoint_id
        )
    else:
        llm_model, llm_tokenizer = load_llm(llm_model_id)
        pairs = generate_romaji_and_chinese(llm_model, llm_tokenizer, lyric_batch)

    aligned: list[AlignedLine] = []
    for index in range(pair_count):
        japanese = lyric_lines[index]
        start = segments[index].start
        if start is None:
            start = 0.0
        start = float(start) + time_offset

        romaji, chinese = pairs[index]
        aligned.append(
            AlignedLine(
                start=start,
                japanese=japanese,
                romaji=romaji,
                chinese=chinese,
            )
        )
        print(f"  [{index + 1}/{pair_count}] {format_lrc_timestamp(start)} {japanese}")

    return aligned


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def write_lrc(path: Path, lines: list[AlignedLine]) -> None:
    content = "\n".join(
        f"{format_lrc_timestamp(line.start)}{line.chinese}" for line in lines
    )
    path.write_text(content + "\n", encoding="utf-8")


def write_txt(path: Path, lines: list[AlignedLine]) -> None:
    blocks: list[str] = []
    for line in lines:
        blocks.append("\n".join([line.romaji, line.japanese, line.chinese]))
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VJ lyrics alignment exporter")
    parser.add_argument("--vocals", required=True, type=Path, help="人声轨，如 vocals.flac")
    parser.add_argument("--lyrics", required=True, type=Path, help="日文歌词 txt，一行一句")
    parser.add_argument("--out-lrc", required=True, type=Path, help="输出纯中文 LRC")
    parser.add_argument("--out-txt", required=True, type=Path, help="输出三语对照 TXT")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="可选本地模型目录，默认 <项目根>/models（有则优先使用）",
    )
    parser.add_argument(
        "--whisper-size",
        default=DEFAULT_WHISPER_SIZE,
        choices=["small", "medium", "large-v2", "large-v3"],
        help="Whisper 模型规格，默认 medium（准确度优先）",
    )
    parser.add_argument("--device", default="cpu", help="推理设备，默认 cpu")
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper compute_type；CPU 用 int8，要更高精度可试 float32",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="可选 Hugging Face Read Token（优先于 hf_token.txt）",
    )
    parser.add_argument(
        "--hf-token-file",
        type=Path,
        default=None,
        help=f"可选 Token 文件路径，默认 <项目根>/{DEFAULT_HF_TOKEN_FILE}",
    )
    parser.add_argument(
        "--llm-backend",
        default=DEFAULT_LLM_BACKEND,
        choices=["doubao", "local"],
        help="LLM 后端：doubao=火山方舟 API，local=本地 transformers",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"本地 LLM 模型（--llm-backend local），默认 {DEFAULT_LLM_MODEL}",
    )
    parser.add_argument(
        "--doubao-endpoint",
        dest="doubao_endpoint",
        default=os.environ.get("ARK_ENDPOINT_ID", DEFAULT_DOUBAO_ENDPOINT),
        help=(
            "火山方舟推理接入点 ID（--llm-backend doubao），"
            f"默认 {DEFAULT_DOUBAO_ENDPOINT}；也可用环境变量 ARK_ENDPOINT_ID"
        ),
    )
    parser.add_argument(
        "--doubao-model",
        dest="doubao_endpoint",
        help="(已废弃，请用 --doubao-endpoint) 同上，传入接入点 ID 而非裸模型名",
    )
    parser.add_argument(
        "--ark-api-key",
        default=None,
        help=f"可选火山方舟 API Key（优先于 {DEFAULT_DOUBAO_TOKEN_FILE}）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_hf_token(cli_token=args.hf_token, token_file=args.hf_token_file)
    sync_hf_token_env()
    configure_hf_endpoint()
    configure_hf_cache()
    if args.llm_backend == "doubao":
        configure_doubao_token(cli_token=args.ark_api_key)
        print(f"[info] 豆包接入点: {args.doubao_endpoint}", flush=True)
    elif args.llm_backend == "local":
        print("[info] LLM 后端: 本地 transformers", flush=True)
    else:
        print(f"[info] LLM 后端: {args.llm_backend}", flush=True)

    try:
        ffmpeg_exe = configure_ffmpeg()
        print(f"[info] 使用 FFmpeg: {ffmpeg_exe}")
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    vocals_path = args.vocals.resolve()
    lyrics_path = args.lyrics.resolve()
    out_lrc = args.out_lrc.resolve()
    out_txt = args.out_txt.resolve()
    models_dir = resolve_models_dir(args.models_dir)

    if not vocals_path.is_file():
        print(f"[error] 找不到人声文件: {vocals_path}", file=sys.stderr)
        return 1
    if not lyrics_path.is_file():
        print(f"[error] 找不到歌词文件: {lyrics_path}", file=sys.stderr)
        return 1

    out_lrc.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    try:
        whisper_ref = ensure_whisper_model(models_dir, args.whisper_size)

        lyric_lines = read_lyric_lines(lyrics_path)
        aligned = align_lyrics_to_audio(
            vocals_path,
            lyric_lines,
            whisper_model_ref=whisper_ref,
            llm_backend=args.llm_backend,
            llm_model_id=args.llm_model,
            doubao_endpoint_id=args.doubao_endpoint,
            device=args.device,
            compute_type=args.compute_type,
        )
        write_lrc(out_lrc, aligned)
        write_txt(out_txt, aligned)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"[done] LRC -> {out_lrc}")
    print(f"[done] TXT -> {out_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
