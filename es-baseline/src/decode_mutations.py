#!/usr/bin/env python3
"""
Decode model responses for the 9 encode/translate SORRY-Bench styles so the
English fine-tuned judge can score them (SORRY-Bench judges the decoded response).

Ciphers (exact, reimplemented to match the encoded prompts):
  caesar (shift 3), atbash (A<->Z), ascii (space-separated decimal codepoints),
  morse (international, words separated by 3 spaces).
Translations: back-translated to English with NLLB-200 (facebook/nllb-200-
distilled-600M). SORRY-Bench uses googletrans; NLLB is a reproducible substitute.

For reasoning models we decode the delivered answer (post-</think>); otherwise the
whole response. Decoding is best-effort (matches SORRY-Bench's unconditional
decode): a response not actually in the cipher decodes to garbage, which the judge
then scores as non-fulfillment.

Reads : results/mutations/<model>__<style>.jsonl
Writes: results/mutations_decoded/<model>__<style>.jsonl  (response replaced)
The 11 direct styles are NOT processed here (judged from the raw responses).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CIPHER_STYLES = ["ascii", "caesar", "morse", "atbash"]
# NLLB source-language codes per translation style
TRANS_LANG = {"translate-fr": "fra_Latn", "translate-zh-cn": "zho_Hans",
              "translate-ml": "mal_Mlym", "translate-mr": "mar_Deva",
              "translate-ta": "tam_Taml"}
TRANS_CHAR_BUDGET = 3000  # translate the lead of long responses

MORSE = {
    ".-": "a", "-...": "b", "-.-.": "c", "-..": "d", ".": "e", "..-.": "f",
    "--.": "g", "....": "h", "..": "i", ".---": "j", "-.-": "k", ".-..": "l",
    "--": "m", "-.": "n", "---": "o", ".--.": "p", "--.-": "q", ".-.": "r",
    "...": "s", "-": "t", "..-": "u", "...-": "v", ".--": "w", "-..-": "x",
    "-.--": "y", "--..": "z", "-----": "0", ".----": "1", "..---": "2",
    "...--": "3", "....-": "4", ".....": "5", "-....": "6", "--...": "7",
    "---..": "8", "----.": "9", ".-.-.-": ".", "--..--": ",", "..--..": "?",
    "-.-.--": "!", "-....-": "-", "-..-.": "/", ".--.-.": "@", "---...": ":",
}


def strip_think(resp: str) -> str:
    if "</think>" in resp:
        return resp.split("</think>", 1)[1]
    return resp


def dec_caesar(t: str, shift: int = 3) -> str:
    out = []
    for c in t:
        if "a" <= c <= "z":
            out.append(chr((ord(c) - 97 - shift) % 26 + 97))
        elif "A" <= c <= "Z":
            out.append(chr((ord(c) - 65 - shift) % 26 + 65))
        else:
            out.append(c)
    return "".join(out)


def dec_atbash(t: str) -> str:
    out = []
    for c in t:
        if "a" <= c <= "z":
            out.append(chr(219 - ord(c)))   # 'a'+'z' = 97+122 = 219
        elif "A" <= c <= "Z":
            out.append(chr(155 - ord(c)))   # 'A'+'Z' = 65+90 = 155
        else:
            out.append(c)
    return "".join(out)


def dec_ascii(t: str) -> str:
    out = []
    for tok in t.split():
        if tok.isdigit():
            try:
                out.append(chr(int(tok)))
                continue
            except (ValueError, OverflowError):
                pass
        out.append(tok)
    return "".join(out)


def dec_morse(t: str) -> str:
    import re
    # letters separated by 1 space, words by 2+ spaces (this dataset uses 2)
    words = []
    for word in re.split(r"\s{2,}", t.replace("/", "  ").strip()):
        letters = "".join(MORSE.get(sym, "") for sym in word.split())
        if letters:
            words.append(letters)
    return " ".join(words)


CIPHER_FN = {"caesar": dec_caesar, "atbash": dec_atbash,
             "ascii": dec_ascii, "morse": dec_morse}


def decode_cipher(style, resp):
    try:
        return CIPHER_FN[style](strip_think(resp))
    except Exception:
        return resp  # best-effort; keep raw on failure


class NLLB:
    def __init__(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        mid = "facebook/nllb-200-distilled-600M"
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            mid, torch_dtype=torch.float16, device_map="auto").eval()
        self.eng_id = self.tok.convert_tokens_to_ids("eng_Latn")

    def translate(self, text, src):
        text = strip_think(text)[:TRANS_CHAR_BUDGET].strip()
        if not text:
            return ""
        self.tok.src_lang = src
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=512).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, forced_bos_token_id=self.eng_id,
                                      max_new_tokens=512)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0]


def process(models, styles, in_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    nllb = None
    need_trans = any(s in TRANS_LANG for s in styles)
    for model in models:
        for style in styles:
            src = in_dir / f"{model}__{style}.jsonl"
            if not src.is_file():
                continue
            dst = out_dir / f"{model}__{style}.jsonl"
            done = set()
            if dst.is_file():
                for l in dst.read_text().split("\n"):
                    try: done.add(json.loads(l)["id"])
                    except Exception: pass
            recs = [json.loads(l) for l in src.read_text().split("\n") if l.strip()]
            todo = [r for r in recs if r["id"] not in done]
            if not todo:
                continue
            if style in TRANS_LANG and nllb is None:
                nllb = NLLB()
            print(f"  decoding {model} / {style}: {len(todo)}")
            with dst.open("a", encoding="utf-8") as w:
                for r in todo:
                    resp = r.get("response") or ""
                    if style in CIPHER_FN:
                        r["response"] = decode_cipher(style, resp)
                    elif style in TRANS_LANG:
                        r["response"] = nllb.translate(resp, TRANS_LANG[style])
                    w.write(json.dumps(r, ensure_ascii=False) + "\n")
    if need_trans and nllb is None:
        print("  (no translation work found)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="opus-4.8,sonnet-5,gpt-5.5,qwen-3b,"
                    "vibethinker-1.5b,deepseek-r1-1.5b,llama-3.2-3b")
    ap.add_argument("--styles", default=",".join(CIPHER_STYLES + list(TRANS_LANG)))
    ap.add_argument("--in-dir", default="./results/mutations")
    ap.add_argument("--out-dir", default="./results/mutations_decoded")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    process(models, styles, Path(args.in_dir), Path(args.out_dir))
    print("Done decoding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
