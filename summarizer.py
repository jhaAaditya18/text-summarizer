# summarizer.py
# Self-contained script to summarize long text (no API keys needed).
# Default behavior: if no args given, it will load "upi.txt"

import argparse
import re
from typing import List
import torch
from transformers import pipeline

MODEL_NAME = "facebook/bart-large-cnn"  # good general-purpose summarizer
DEFAULT_INPUT_FILE = "upi.txt"          # 👈 default file


def clean_text(t: str) -> str:
    t = t.replace("\u2003", " ").replace("\u2002", " ").replace("\u2009", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    return t.strip()


def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    if len(parts) <= 1:
        parts = re.split(r"(?<=[.;:])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_sentences(sents: List[str], max_chars: int = 2800, overlap_chars: int = 200) -> List[str]:
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        if cur_len + len(s) + 1 > max_chars:
            if cur:
                chunks.append(" ".join(cur))
                if overlap_chars > 0 and chunks[-1]:
                    tail = chunks[-1][-overlap_chars:]
                    cur = [tail]
                    cur_len = len(tail)
                else:
                    cur, cur_len = [], 0
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def make_summarizer():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "summarization",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=device,
    )


def summarize_chunk(summarizer, text: str, min_len: int = 80, max_len: int = 220) -> str:
    out = summarizer(text, min_length=min_len, max_length=max_len, do_sample=False, truncation=True)
    return out[0]["summary_text"].strip()


def map_reduce_summarize(text: str, style: str = "short") -> str:
    text = clean_text(text)
    sents = split_into_sentences(text)
    chunks = chunk_sentences(sents, max_chars=2800, overlap_chars=220)

    summarizer = make_summarizer()

    first_pass = []
    for ch in chunks:
        fp = summarize_chunk(summarizer, ch, min_len=80, max_len=200)
        first_pass.append(fp)

    combined = " ".join(first_pass)

    if style == "short":
        return summarize_chunk(summarizer, combined, min_len=80, max_len=170)

    if style == "detailed":
        mid = summarize_chunk(summarizer, combined, min_len=120, max_len=260)
        return summarize_chunk(summarizer, mid, min_len=140, max_len=300)

    if style == "bullets":
        dense = summarize_chunk(summarizer, combined, min_len=120, max_len=260)
        pts = re.split(r"(?<=[.!?])\s+", dense)
        pts = [p.strip(" -•") for p in pts if len(p.strip()) > 0]
        pts = pts[:8]
        bullets = []
        for p in pts:
            if len(p) > 220:
                p = summarize_chunk(make_summarizer(), p, min_len=40, max_len=70)
            bullets.append(f"• {p}")
        return "\n".join(bullets)

    return summarize_chunk(summarizer, combined, min_len=80, max_len=170)


def main():
    ap = argparse.ArgumentParser(description="Summarize long text with chunked map-reduce using BART.")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--input", "-i", type=str, help="Path to a UTF-8 text file to summarize")
    src.add_argument("--text", "-t", type=str, help="Raw text to summarize (quote it)")
    ap.add_argument("--style", choices=["short", "detailed", "bullets"], default="short", help="Summary style")
    args = ap.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        # 👇 default behavior: read from upi.txt
        with open(DEFAULT_INPUT_FILE, "r", encoding="utf-8") as f:
            text = f.read()

    summary = map_reduce_summarize(text, style=args.style)
    print(summary)


if __name__ == "__main__":
    main()
