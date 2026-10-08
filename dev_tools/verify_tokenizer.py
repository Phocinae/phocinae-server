"""Differential test: phocinae GemmaTokenizer vs the reference `tokenizers` build.

Loads the same tokenizer.json both ways and asserts identical encode() output
(and decode round-trips) over a corpus covering ASCII/unicode/newlines/tabs/
CJK/emoji/added-token runs/long words.

Usage: /home/hermes/decision-model/.venv/bin/python dev_tools/verify_tokenizer.py
"""

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from phocinae.tokenizer import GemmaTokenizer  # noqa: E402

MODEL_DIR = "/home/hermes/decision-model/09_常态探索/release_prep_20261007/02_hf_release/hf_repo"
TOK_DIR = os.path.join(MODEL_DIR, "tokenizer")


def load_reference():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(TOK_DIR, "tokenizer.json"))


def main():
    mine = GemmaTokenizer(TOK_DIR)
    ref = load_reference()

    corpus = [
        "", " ", "  ", "hello", "hello world", "the the the", "a", "b", "a\nb",
        "a \nb", "a\n\nb", "a\n\tb", "line one\nline two\n\nline three",
        "\t\tindented", "  two spaces", " leading", "trailing ",
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world! This is a test of the emergency broadcast system.",
        "中文测试", "中文 测试", "混合 mixed 文本 text", "🚀 emoji test 🎉",
        "under_score snake_case camelCase PascalCase kebab-case",
        "hyphenated-word supercalifragilisticexpialidocious",
        "x\u2028y", "tab\there", "numbers 12345 3.14159",
        "<h1>heading</h1>", "[toxicity=0] low tox [toxicity=1] high",
        "<unused10> token", "(@BOS@)", "@BOS@x",
        "a" * 200, "字" * 100, ("word " * 60).strip(),
        "multi\n\n\n\nnewline runs \t\t\t tabs  ▁▁ metaspace chars",
        "One\nTwo\nThree\nFour\nFive\nSix\nSeven\nEight\nNine\nTen",
        "don't can't won't it's I'm you're they've we'll",
        "state: The agent wrote to /home/user/file.txt and restarted nginx.",
    ]
    # random junk: mixed ascii/unicode with occasional newlines
    rng = random.Random(42)
    alphabet = "ab cd\nef\t0123 XYZ中文▁🚀<h1>\n\n-_.:;/"
    for _ in range(40):
        n = rng.randint(0, 300)
        corpus.append("".join(rng.choice(alphabet) for _ in range(n)))
    # round-trip decode corpus from real td rows' states
    td_rows = 0
    with open("/home/hermes/decision-model/exp/typed_decisions_test.jsonl",
              encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            corpus.append(r.get("state", ""))
            td_rows += 1
            if td_rows >= 25:
                break

    bad = 0
    for i, text in enumerate(corpus):
        got = mine.encode(text)
        want = ref.encode(text, add_special_tokens=False).ids
        if got != want:
            bad += 1
            print("MISMATCH #%d %r\n  mine: %s\n  ref : %s" % (i, text[:80], got[:40], want[:40]))
            if bad >= 5:
                break
        # decode round-trip on a sample of texts
        if i % 7 == 0 and got:
            mine_rt = mine.decode(got)
            ref_rt = ref.decode(got)
            if mine_rt != ref_rt:
                bad += 1
                print("DECODE MISMATCH %r\n  mine: %r\n  ref : %r" % (text[:80], mine_rt[:60], ref_rt[:60]))
    print("checked %d texts (+%d td states), mismatches: %d" % (len(corpus), td_rows, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
