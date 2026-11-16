"""Vendor chapter 0 embedded images/plots and keep regenerated notebooks local.

Run --fetch only when adding/updating upstream media; normal use needs no downloads.
Source URLs and SHA-256 hashes are retained in the checked-in manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "chapter0_fundamentals"
MANIFEST = CHAPTER / "assets" / "manifest.json"
URL = r'https?://[^\s<>"\'\)\]`]+'
IMAGE = re.compile(
    r'<img\b[^>]*?src\s*=\s*["\'](' + URL + r')["\']|!\[[^\]]*\]\((' + URL + r')\)', re.IGNORECASE
)
IFRAME = re.compile(
    r'<iframe\b[^>]*?src\s*=\s*["\'](' + URL + r')["\'][^>]*>\s*</iframe>', re.IGNORECASE
)
RESOURCE = re.compile(r'(?:src|href)\s*=\s*["\'](' + URL + r')["\']', re.IGNORECASE)


def embedded_urls(text: str) -> set[str]:
    images = {a or b for a, b in IMAGE.findall(text)}
    plots = {
        u
        for u in re.findall(
            r'<(?:iframe|embed|video|source)\b[^>]*?src\s*=\s*["\'](' + URL + r')["\']',
            text,
            re.IGNORECASE,
        )
        if urlsplit(u).hostname == "info-arena.github.io"
    }
    return images | plots


def read_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []


def asset_path(url: str) -> str:
    name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    name = re.sub(r'[^\w.\-]', '_', name)[:100] or "image"
    if not Path(name).suffix:
        name += ".jpg" if "format=jpg" in url else ".png"
    return "assets/" + hashlib.sha256(url.encode()).hexdigest()[:12] + "-" + name


def fetch_one(url: str) -> dict:
    relative = asset_path(url)
    path = CHAPTER / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    # GitHub's blob page redirects inconsistently through proxies; request raw bytes directly.
    download_url = url
    if url.startswith("https://github.com/") and "/blob/" in url:
        download_url = (
            url.replace("github.com/", "raw.githubusercontent.com/")
            .replace("/blob/", "/")
            .split("?")[0]
        )
    if url.startswith("https://info-arena.github.io/ARENA_img/"):
        download_url = url.replace(
            "https://info-arena.github.io/ARENA_img/",
            "https://raw.githubusercontent.com/info-arena/ARENA_img/main/",
        )
    temp = path.with_suffix(path.suffix + ".tmp")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "2",
            "--max-time",
            "90",
            download_url,
            "--output",
            str(temp),
        ],
        check=True,
    )
    data = temp.read_bytes()
    if not data or (path.suffix != ".html" and b"<html" in data[:500].lower()):
        raise ValueError(f"Not an image: {url}")
    temp.replace(path)
    return {"url": url, "path": relative, "sha256": hashlib.sha256(data).hexdigest()}


def fetch() -> None:
    entries = {entry["url"]: entry for entry in read_manifest()}
    urls = set()
    for path in (ROOT / "infrastructure/chapters/chapter0_fundamentals").rglob("master_*.py"):
        urls.update(embedded_urls(path.read_text()))
    pending = sorted(urls - entries.keys())
    failures = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [(url, pool.submit(fetch_one, url)) for url in pending]
        for url, future in futures:
            try:
                entry = future.result()
                entries[url] = entry
                print(f"Saved {entry['path']}", flush=True)
            except (subprocess.CalledProcessError, OSError, ValueError) as error:
                failures.append(url)
                print(f"FAILED {url}: {error}", flush=True)
    # Plotly HTML embeds also need their JS locally. Replace external script resources.
    for entry in list(entries.values()):
        path = CHAPTER / entry["path"]
        if path.suffix != ".html":
            continue
        html = path.read_text()
        for url in RESOURCE.findall(html):
            if url not in entries:
                entries[url] = fetch_one(url)
            html = html.replace(url, os.path.relpath(CHAPTER / entries[url]["path"], path.parent))
        path.write_text(html)
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(sorted(entries.values(), key=lambda e: e["url"]), indent=2) + "\n"
    )
    if failures:
        raise SystemExit(f"Failed to download {len(failures)} asset(s); manifest saved for retry.")


def localize_text(text: str, parent: Path) -> str:
    for entry in read_manifest():
        if entry["url"] in text:
            relative = Path(os.path.relpath(CHAPTER / entry["path"], parent)).as_posix()
            text = text.replace(entry["url"], relative)

    # Videos are optional external reading, not auto-loaded embeds.
    def video_link(match):
        url = match.group(1)
        return (
            f"[Watch the optional lecture (requires internet)]({url})"
            if "youtube.com" in url
            else match.group(0)
        )

    return IFRAME.sub(video_link, text)


def localize_file(path: Path) -> None:
    old = path.read_text()
    if path.suffix == ".ipynb":
        notebook = json.loads(old)
        for cell in notebook["cells"]:
            text = "".join(cell["source"])
            cell["source"] = localize_text(text, path.parent).splitlines(keepends=True)
        new = json.dumps(notebook)
    else:
        new = localize_text(old, path.parent)
    if old != new:
        path.write_text(new)


def check() -> None:
    entries = read_manifest()
    assert entries, "No asset manifest; run scripts/chapter0_assets.py --fetch"
    for entry in entries:
        path = CHAPTER / entry["path"]
        assert path.is_file(), f"Missing asset: {path}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], (
            f"Changed asset: {path}"
        )
        if path.suffix == ".html":
            assert not RESOURCE.findall(path.read_text()), f"Remote dependency in {path}"
    notebooks = list((CHAPTER / "exercises").rglob("*.ipynb"))
    for path in notebooks:
        for cell in json.loads(path.read_text())["cells"]:
            text = "".join(cell["source"])
            assert not embedded_urls(text), f"Remote image/plot in {path}"
            assert not IFRAME.findall(text) or all(
                not u.startswith("http") for u in IFRAME.findall(text)
            ), f"Remote iframe in {path}"
            for relative in re.findall(r'(?:\.\./)+assets/[\w.\-]+', text):
                assert (path.parent / relative).is_file(), f"Missing reference {relative} in {path}"
    print(f"Verified {len(entries)} local assets and {len(notebooks)} notebooks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--localize", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch()
    if args.localize:
        for path in (CHAPTER / "exercises").rglob("*.ipynb"):
            localize_file(path)
    if args.check:
        check()
