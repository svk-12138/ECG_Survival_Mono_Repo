#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计 XML 文件中指定标签（默认 StmtText）含特定中文关键字的次数（支持模糊匹配）。"""
import argparse
import base64
import binascii
import io
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import xml.etree.ElementTree as ET

DEFAULT_XML_DIR = Path("/home/admin123/use/Program/ECG_Survival_Mono_Repo/data/XML")
DEFAULT_KEYWORDS = ("房性", "室性", "起搏", "窦性心律")
DEFAULT_THRESHOLD = 0.8
DEFAULT_TARGET_TAG = "StmtText"


def parse_args():
    parser = argparse.ArgumentParser(
        description="遍历 XML，统计指定标签中包含关键字的出现次数"
    )
    parser.add_argument(
        "--xml_dir",
        type=Path,
        default=DEFAULT_XML_DIR,
        help="XML 文件目录（递归扫描）",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=list(DEFAULT_KEYWORDS),
        help="需要匹配的关键字，默认：房性 室性 起搏",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=DEFAULT_TARGET_TAG,
        help="需要检索的标签名，默认 StmtText（若需匹配 StmtFlag 可切换）。",
    )
    parser.add_argument(
        "--fuzzy_threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="模糊匹配阈值 (0-1)。<=0 表示仅精确匹配，默认 0.8。",
    )
    return parser.parse_args()


def _tag_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _load_xml_bytes(xml_path: Path) -> bytes:
    """读取 XML 文件，若内容整体为 base64 则自动解码。"""
    data = xml_path.read_bytes()
    if b"<" in data[:200]:
        return data
    compact = b"".join(data.split())
    if len(compact) >= 8 and len(compact) % 4 == 0:
        base64_chars = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
        if all(c in base64_chars for c in compact):
            try:
                decoded = base64.b64decode(compact, validate=True)
                if b"<" in decoded:
                    return decoded
            except (binascii.Error, ValueError):
                pass
    raise RuntimeError(f"{xml_path} 不像有效 XML，也无法识别为 base64。")


def _maybe_decode_base64(text: str) -> str:
    """尝试将 StmtText 中的 base64 内容还原为可读字符串。"""
    if not text:
        return text
    compact = "".join(text.split())
    if len(compact) < 8 or len(compact) % 4 != 0:
        return text
    base64_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    if any(c not in base64_chars for c in compact):
        return text
    try:
        decoded = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError):
        return text
    for enc in ("utf-8", "gbk", "latin1"):
        try:
            return decoded.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    return decoded.decode("utf-8", errors="ignore").strip()


def _fuzzy_contains(text: str, keyword: str, threshold: float, matcher: SequenceMatcher) -> bool:
    if not text or not keyword:
        return False
    if threshold <= 0:
        return keyword in text
    if keyword in text:
        return True
    text_len = len(text)
    kw_len = len(keyword)
    max_extra = max(1, int(kw_len * 0.5))
    for start in range(0, max(text_len - kw_len + 1, 1)):
        end = min(text_len, start + kw_len + max_extra)
        candidate = text[start:end]
        if not candidate:
            continue
        matcher.set_seqs(keyword, candidate)
        if matcher.ratio() >= threshold:
            return True
    return False


def scan_stmt_texts(xml_path: Path, keywords: list[str], threshold: float, target_tag: str) -> Counter:
    counts = Counter()
    matcher = SequenceMatcher()
    target_tag = target_tag or DEFAULT_TARGET_TAG
    try:
        xml_bytes = _load_xml_bytes(xml_path)
        buffer = io.BytesIO(xml_bytes)
        for _, elem in ET.iterparse(buffer, events=("end",)):
            if _tag_name(elem.tag) == target_tag:
                raw_text = (elem.text or "").strip()
                text = _maybe_decode_base64(raw_text)
                for kw in keywords:
                    if _fuzzy_contains(text, kw, threshold, matcher):
                        counts[kw] += 1
                elem.clear()
    except Exception as exc:
        raise RuntimeError(f"解析 {xml_path} 失败: {exc}") from exc
    return counts


def main():
    args = parse_args()
    xml_dir = args.xml_dir
    keywords = list(dict.fromkeys(args.keywords))  # 去重保持顺序
    threshold = args.fuzzy_threshold
    target_tag = args.tag
    if not xml_dir.exists():
        raise FileNotFoundError(f"XML 目录不存在: {xml_dir}")

    total_files = 0
    counts = Counter()
    file_hits = defaultdict(int)
    errors = []

    for xml_file in sorted(xml_dir.rglob("*.xml")):
        total_files += 1
        try:
            file_counts = scan_stmt_texts(xml_file, keywords, threshold, target_tag)
        except RuntimeError as err:
            errors.append(str(err))
            continue
        for kw in keywords:
            kw_hits = file_counts.get(kw, 0)
            if kw_hits:
                counts[kw] += kw_hits
                file_hits[kw] += 1

    print(f"扫描 {total_files} 个 XML 文件，目标标签 <{target_tag}>")
    for kw in keywords:
        print(f"关键字[{kw}]：匹配 {counts[kw]} 次，涉及 {file_hits[kw]} 个文件")
    if errors:
        print(f"\n共有 {len(errors)} 个文件解析失败，前 5 条错误：")
        for msg in errors[:5]:
            print(" -", msg)


if __name__ == "__main__":
    main()
