#!/usr/bin/env python3
import hashlib
import io
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(".").resolve()
EXCLUDE = [
    ".git/",
    ".venv",
    "__pycache__",
    "node_modules/",
    "dist/",
    "build/",
    ".tox/",
    ".mypy_cache/",
    ".idea/",
    ".vscode/",
    ".cache/",
]
BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".onnx",
    ".pth",
    ".pt",
    ".bin",
    ".wav",
    ".mp3",
    ".flac",
}
ISS = []


def add(t, p, d):
    ISS.append((t, str(p), d))


def skip(p: Path):
    s = str(p).replace("\\", "/") + ("/" if p.is_dir() else "")
    return any(x in s for x in EXCLUDE)


# Collect files with progress
files = []
t0 = time.time()
for dp, dn, fn in os.walk(ROOT):
    dpp = Path(dp)
    if skip(dpp):
        dn[:] = []
        continue
    for n in fn:
        fp = dpp / n
        if not skip(fp):
            files.append(fp)
            if len(files) % 500 == 0:
                print(f"[scan] indexed {len(files)} files in {time.time() - t0:.1f}s", flush=True)

print(f"[scan] total files: {len(files)}", flush=True)

# Quick integrity checks with periodic progress
for i, f in enumerate(files, 1):
    try:
        sz = f.stat().st_size
        if sz == 0:
            add("Missing/Empty", f, "Zero-byte file")
        with open(f, "rb") as r:
            chunk = r.read(4096)
            if b"\x00" in chunk and f.suffix.lower() not in BINARY_SUFFIXES:
                add("Binary/Unexpected", f, "NUL bytes found (binary in text tree?)")
    except Exception as e:
        add("Unreadable", f, f"stat/open failed: {e}")

    # Text-level heuristics
    try:
        txt = f.read_text(encoding="utf-8")
        lines = txt.splitlines(keepends=True)
        if txt and not txt.endswith(("\n", "\r", "\r\n")):
            add("Truncation-Heuristic", f, "No trailing newline at EOF")
        if any(re.match(r"^(<{7}|={7}|>{7})", ln.rstrip("\r\n")) for ln in lines):
            add("Merge-Conflict", f, "Git conflict markers present")
        for j, ln in enumerate(lines, 1):
            if re.search(r"(TRUNCATED|INCOMPLETE|TODO|FIXME)", ln, re.I):
                add("Flagged-Note", f, f"Line {j}: {ln.strip()[:220]}")
            if len(ln) > 2000:
                add("Suspicious-Line", f, f"Line {j}: >2000 chars")
    except UnicodeDecodeError as e:
        add("Encoding", f, f"Not UTF-8 decodable: {e}")
    except Exception as e:
        add("Unreadable", f, f"text read failed: {e}")

    if i % 500 == 0:
        print(f"[check] processed {i}/{len(files)} files", flush=True)

# JSON / YAML checks (safe if libs missing)
for f in [p for p in files if p.suffix.lower() in {".json", ".jsonc"}]:
    try:
        raw = f.read_text(encoding="utf-8")
        if f.suffix.lower() == ".jsonc":
            raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
            raw = re.sub(r"^\s*//.*?$", "", raw, flags=re.M)
        json.loads(raw)
    except Exception as e:
        add("JSON", f, f"Invalid JSON: {e}")

try:
    import yaml  # type: ignore

    for f in [p for p in files if p.suffix.lower() in {".yml", ".yaml"}]:
        try:
            yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception as e:
            add("YAML", f, f"Invalid YAML: {e}")
except Exception:
    add("YAML", ROOT / "(global)", "PyYAML not installed — YAML validation skipped")

# Markdown relative links
MD_LINK = re.compile(r"\[[^\]]+\]\((?!https?://|mailto:)([^)\s#]+)")
for f in [p for p in files if p.suffix.lower() == ".md"]:
    try:
        txt = f.read_text(encoding="utf-8")
        for m in MD_LINK.finditer(txt):
            tgt = (f.parent / m.group(1)).resolve()
            if not tgt.exists():
                add("MD-Link", f, f"Broken relative link: {m.group(1)}")
    except Exception as e:
        add("Unreadable", f, f"MD read failed: {e}")

# Duplicate content hash
hmap = {}
for f in files:
    try:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        hmap.setdefault(h, []).append(f)
    except Exception as e:
        add("Unreadable", f, f"hash failed: {e}")
for h, ps in hmap.items():
    if len(ps) > 1:
        add("Duplicate-Content", ps[0], "Also seen in: " + "; ".join(map(str, ps)))

# Write report
rep = ROOT / "repo_scan_report.md"
groups = defaultdict(list)
for t, p, d in ISS:
    groups[t].append((p, d))
buf = io.StringIO()
buf.write(f"## Repo Scan Report ({ROOT})\n\nFiles scanned: {len(files)}\n")
for t in sorted(groups.keys()):
    buf.write(f"\n### {t}  ({len(groups[t])})\n")
    for p, d in sorted(groups[t], key=lambda x: (x[0], x[1]))[:500]:
        buf.write(f"* **{p}** — {d}\n")
rep.write_text(buf.getvalue(), encoding="utf-8")
print("[done] Scan complete -> repo_scan_report.md", flush=True)
