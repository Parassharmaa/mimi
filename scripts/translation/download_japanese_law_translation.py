#!/usr/bin/env python3
"""Download finalized Japanese Law Translation TMX files with provenance."""

from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ROOT = "https://www.japaneselawtranslation.go.jp"
SEARCH_URL = f"{ROOT}/en/laws"
RESULT_URL = f"{ROOT}/en/laws/result/"
TERMS_URL = f"{ROOT}/en/index/terms"
PDL_URL = "https://www.digital.go.jp/en/resources/open_data/public_data_license_v1.0"
USER_AGENT = "Mimi-Translation-Research/1.0 (license and corpus audit)"
RESULT_COUNT_RE = re.compile(r"Showing\s+\d+\s+to\s+\d+\s+of\s+(\d+)")
LAW_VIEW_RE = re.compile(r"^/en/laws/view/(\d+)$")


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.csrf_token: str | None = None
        self.law_ids: set[str] = set()
        self.tmx_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "input" and attributes.get("name") == "_csrfToken":
            self.csrf_token = attributes.get("value")
        if tag != "a" or not attributes.get("href"):
            return
        href = attributes["href"]
        assert href is not None
        match = LAW_VIEW_RE.fullmatch(href)
        if match:
            self.law_ids.add(match.group(1))
        if attributes.get("id") == "tmxDownload":
            self.tmx_href = href


def decode_html(raw: bytes) -> str:
    # A few legacy law views contain isolated non-UTF-8 bytes even though the site
    # declares UTF-8. Replacement is safe here because all extracted URLs/IDs and
    # form field names are ASCII; TMX payloads remain strict XML below.
    return raw.replace(b"\0", b"").decode("utf-8", errors="replace")


def parse_search_page(raw: bytes) -> tuple[set[str], int]:
    text = decode_html(raw)
    parser = LinkParser()
    parser.feed(text)
    match = RESULT_COUNT_RE.search(text)
    if match is None:
        raise SystemExit("Japanese Law Translation search result lacks result count")
    return parser.law_ids, int(match.group(1))


def parse_csrf(raw: bytes) -> str:
    parser = LinkParser()
    parser.feed(decode_html(raw))
    if not parser.csrf_token:
        raise SystemExit("Japanese Law Translation search form lacks CSRF token")
    return parser.csrf_token


def parse_tmx_href(raw: bytes) -> str | None:
    parser = LinkParser()
    parser.feed(decode_html(raw))
    return parser.tmx_href


class Downloader:
    def __init__(self, *, timeout: float, retries: int, delay_seconds: float) -> None:
        cookie_jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cookie_jar)
        )
        self.timeout = timeout
        self.retries = retries
        self.delay_seconds = delay_seconds

    def request(self, url: str, data: dict[str, str] | None = None) -> bytes:
        encoded = urllib.parse.urlencode(data).encode("utf-8") if data else None
        request = urllib.request.Request(
            url,
            data=encoded,
            headers={"User-Agent": USER_AGENT},
            method="POST" if data else "GET",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    raw = response.read()
                if self.delay_seconds:
                    time.sleep(self.delay_seconds)
                return raw
            except urllib.error.HTTPError as error:
                last_error = error
                if attempt == self.retries:
                    break
                if error.code in {403, 408, 429, 500, 502, 503, 504}:
                    retry_after = error.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else min(30.0, 5.0 * (2 ** attempt))
                else:
                    wait = min(8.0, 0.5 * (2 ** attempt))
                time.sleep(wait)
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
                if attempt == self.retries:
                    break
                time.sleep(min(8.0, 0.5 * (2 ** attempt)))
        raise SystemExit(f"download failed after retries: {url}: {last_error}")


def search_form(csrf_token: str, letter: str) -> dict[str, str]:
    # Omitting ia=03 is intentional: tentative translations are excluded.
    return {"_csrfToken": csrf_token, "al[]": letter, "amb": "on"}


def tmx_metadata(raw: bytes, label: str) -> dict[str, Any]:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise ValueError(f"not well-formed XML: {error}") from error
    header = root.find("header")
    body = root.find("body")
    if root.tag != "tmx" or header is None or body is None:
        raise ValueError(f"downloaded file is not TMX ({label})")
    return {
        "creationdate": header.attrib.get("creationdate"),
        "creationid": header.attrib.get("creationid"),
        "source_language": header.attrib.get("srclang"),
        "translation_units": len(body.findall("tu")),
    }


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--letters", default="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    parser.add_argument("--max-laws", type=int)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=0.20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()) and not args.resume:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output_directory / "inventory.json"
    prior: dict[str, Any] = {}
    if args.resume and inventory_path.is_file():
        prior = json.loads(inventory_path.read_text(encoding="utf-8"))
    downloader = Downloader(
        timeout=args.timeout, retries=args.retries, delay_seconds=args.delay_seconds
    )
    prior_ids = prior.get("discovered_law_ids")
    if isinstance(prior_ids, list) and all(str(value).isdigit() for value in prior_ids):
        law_ids = {str(value) for value in prior_ids}
        search_records = list(prior.get("search_records", []))
    else:
        csrf_token = parse_csrf(downloader.request(SEARCH_URL))
        law_ids = set()
        search_records = []
        for letter in dict.fromkeys(args.letters.upper()):
            data = search_form(csrf_token, letter)
            first_raw = downloader.request(RESULT_URL, data)
            first_ids, total = parse_search_page(first_raw)
            letter_ids = set(first_ids)
            pages = math.ceil(total / 50)
            for page in range(2, pages + 1):
                page_raw = downloader.request(f"{RESULT_URL}?page={page}", data)
                page_ids, page_total = parse_search_page(page_raw)
                if page_total != total:
                    raise SystemExit(f"result count changed while crawling letter {letter}")
                letter_ids.update(page_ids)
            law_ids.update(letter_ids)
            search_records.append(
                {
                    "letter": letter,
                    "reported_results": total,
                    "unique_law_ids": len(letter_ids),
                }
            )

    ordered_ids = sorted(law_ids, key=int)
    if args.max_laws is not None:
        ordered_ids = ordered_ids[: args.max_laws]
    records_by_id = {
        str(record["law_id"]): record
        for record in prior.get("files", [])
        if isinstance(record, dict) and record.get("law_id")
    }
    missing_tmx: set[str] = set(prior.get("missing_tmx", []))
    rejected_by_id = {
        str(record["law_id"]): record
        for record in prior.get("rejected_tmx", [])
        if isinstance(record, dict) and record.get("law_id")
    }

    def inventory() -> dict[str, Any]:
        files = [records_by_id[key] for key in sorted(records_by_id, key=int)]
        accounted = set(records_by_id) | missing_tmx | set(rejected_by_id)
        unaccounted = sorted(set(ordered_ids) - accounted, key=int)
        return {
            "schema_version": 1,
            "source": SEARCH_URL,
            "terms_url": TERMS_URL,
            "public_data_license_url": PDL_URL,
            "license": "PDL-1.0-compatible-CC-BY-4.0",
            "tentative_translations_included": False,
            "search_letters": list(dict.fromkeys(args.letters.upper())),
            "search_records": search_records,
            "discovered_law_ids": sorted(law_ids, key=int),
            "discovered_unique_laws": len(law_ids),
            "selected_laws": len(ordered_ids),
            "complete": not unaccounted,
            "unaccounted_law_ids": unaccounted,
            "files": files,
            "missing_tmx": sorted(missing_tmx, key=int),
            "rejected_tmx": [
                rejected_by_id[key] for key in sorted(rejected_by_id, key=int)
            ],
            "downloaded_files": len(files),
            "downloaded_bytes": sum(record["bytes"] for record in files),
            "downloaded_translation_units": sum(
                record["translation_units"] for record in files
            ),
        }

    atomic_json(inventory_path, inventory())
    for law_id in ordered_ids:
        if law_id in missing_tmx:
            continue
        existing = records_by_id.get(law_id)
        if existing:
            path = args.output_directory / existing["filename"]
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == existing["sha256"]:
                continue
        rejected = rejected_by_id.get(law_id)
        if rejected:
            path = args.output_directory / rejected["filename"]
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == rejected["sha256"]:
                continue
        view_url = f"{SEARCH_URL}/view/{law_id}"
        href = parse_tmx_href(downloader.request(view_url))
        if href is None:
            missing_tmx.add(law_id)
            atomic_json(inventory_path, inventory())
            continue
        tmx_url = urllib.parse.urljoin(ROOT, href)
        if urllib.parse.urlparse(tmx_url).netloc != urllib.parse.urlparse(ROOT).netloc:
            raise SystemExit(f"refusing cross-host TMX URL: {tmx_url}")
        raw = downloader.request(tmx_url)
        try:
            metadata = tmx_metadata(raw, law_id)
        except ValueError as error:
            rejected_filename = f"law-{law_id}.tmx.rejected"
            (args.output_directory / rejected_filename).write_bytes(raw)
            rejected_by_id[law_id] = {
                "law_id": law_id,
                "filename": rejected_filename,
                "view_url": view_url,
                "tmx_url": tmx_url,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "reason": str(error),
            }
            atomic_json(inventory_path, inventory())
            continue
        filename = f"law-{law_id}.tmx"
        path = args.output_directory / filename
        path.write_bytes(raw)
        records_by_id[law_id] = {
            "law_id": law_id,
            "filename": filename,
            "view_url": view_url,
            "tmx_url": tmx_url,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            **metadata,
        }
        atomic_json(inventory_path, inventory())

    result = inventory()
    atomic_json(inventory_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
