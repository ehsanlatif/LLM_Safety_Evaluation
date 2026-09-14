#!/usr/bin/env python3
"""
Download safety benchmark datasets to a local directory.

Datasets & sources (they live in DIFFERENT places):
  1. AILuminate (MLCommons) DEMO prompt set - 1,200 prompts across 12 hazard
     categories, CC BY 4.0. Hosted on GitHub, NOT Hugging Face.
     https://github.com/mlcommons/ailuminate
  2. SORRY-Bench (Xie et al., ICLR 2025) - academic refusal benchmark,
     44 fine-grained risk categories. Hosted on Hugging Face.
     https://huggingface.co/datasets/sorry-bench/sorry-bench-202503

Only dependency: requests  (pip install requests)
No huggingface_hub needed - SORRY-Bench files are enumerated via the public
Hugging Face API and pulled through their resolve URLs.

Usage:
    python3 download_datasets.py [--out-dir ./data]

Auth:
  - The Hugging Face token is read (in priority order) from --token, the
    HF_TOKEN / HF_Token env vars, or a .env file next to this script.
    Recognized .env keys: HF_TOKEN, HF_Token, HUGGINGFACE_TOKEN,
    HUGGINGFACE_HUB_TOKEN (case-insensitive).
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Missing dependency 'requests'. Install with:  pip install requests")


# --- AILuminate: individual files fetched via GitHub raw URLs ---------------
AILUMINATE_RAW_BASE = "https://raw.githubusercontent.com/mlcommons/ailuminate/main"
AILUMINATE_REQUIRED = "airr_official_1.0_demo_en_us_prompt_set_release.csv"
AILUMINATE_FILES = [AILUMINATE_REQUIRED, "README.md", "LICENSE.md"]

# --- SORRY-Bench: enumerated from the Hugging Face dataset API ---------------
SORRYBENCH_REPO = "sorry-bench/sorry-bench-202503"

# Env / .env keys that may hold a Hugging Face token (matched case-insensitively).
HF_TOKEN_KEYS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")


def resolve_token(cli_token: str | None) -> str | None:
    """Find an HF token from --token, environment, or a local .env file."""
    if cli_token:
        return cli_token
    # Environment (case-insensitive match against known key names).
    for key, value in os.environ.items():
        if key.upper() in HF_TOKEN_KEYS and value.strip():
            return value.strip()
    # .env file at the repo root (src/ is one level down).
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export").strip()
            if key.upper() in HF_TOKEN_KEYS:
                return value.strip().strip("'\"")
    return None


def _save(resp: requests.Response, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            fh.write(chunk)


def download_ailuminate(target_dir: Path) -> None:
    print(f"\n=== Downloading AILuminate (GitHub) -> {target_dir} ===")
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in AILUMINATE_FILES:
        url = f"{AILUMINATE_RAW_BASE}/{name}"
        print(f"    fetching {name} ...")
        resp = requests.get(url, timeout=120, stream=True)
        if resp.status_code == 404 and name != AILUMINATE_REQUIRED:
            print(f"    (skipped, not found: {name})")
            continue
        resp.raise_for_status()
        _save(resp, target_dir / name)
        print(f"    saved {target_dir / name}")


def download_sorrybench(target_dir: Path, token: str | None) -> None:
    print(f"\n=== Downloading SORRY-Bench (Hugging Face) -> {target_dir} ===")
    target_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    api = f"https://huggingface.co/api/datasets/{SORRYBENCH_REPO}"
    meta = requests.get(api, headers=headers, timeout=60)
    meta.raise_for_status()
    info = meta.json()
    files = [s["rfilename"] for s in info.get("siblings", [])]
    if not files:
        raise RuntimeError("no files listed for SORRY-Bench repo")

    if info.get("gated") and not token:
        raise RuntimeError(_gated_help())

    for name in files:
        url = f"https://huggingface.co/datasets/{SORRYBENCH_REPO}/resolve/main/{name}"
        print(f"    fetching {name} ...")
        resp = requests.get(url, headers=headers, timeout=300, stream=True)
        if resp.status_code in (401, 403):
            raise RuntimeError(_gated_help())
        resp.raise_for_status()
        _save(resp, target_dir / name)
    print(f"    saved {len(files)} files")


def _gated_help() -> str:
    return (
        f"SORRY-Bench is a GATED Hugging Face dataset. To download it:\n"
        f"      1. Sign in at https://huggingface.co/datasets/{SORRYBENCH_REPO}\n"
        f"         and click 'Agree and access repository' to accept the terms.\n"
        f"      2. Create a token at https://huggingface.co/settings/tokens\n"
        f"      3. Re-run with:  HF_TOKEN=hf_xxx python3 download_datasets.py\n"
        f"         (or pass --token hf_xxx)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default="./data", help="Download directory (default: ./data)"
    )
    parser.add_argument(
        "--token", default=None, help="Hugging Face token (falls back to HF_TOKEN)."
    )
    args = parser.parse_args()

    token = resolve_token(args.token)
    print(f"Hugging Face token: {'found' if token else 'not found'}")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("AILuminate", lambda: download_ailuminate(out_dir / "ailuminate")),
        ("SORRY-Bench", lambda: download_sorrybench(out_dir / "sorry-bench", token)),
    ]

    failures = []
    for name, fn in jobs:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"    FAILED: {name}: {exc}", file=sys.stderr)

    print("\n=== Summary ===")
    for name, _ in jobs:
        print(f"  [{'FAILED' if name in failures else 'OK'}] {name}")

    if failures:
        return 1
    print(f"\nAll datasets downloaded to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
