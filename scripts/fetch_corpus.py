"""Fetch the real public-document corpus described in data/corpus_manifest.csv.

Documents are never vendored (see claude.md): only the manifest is committed,
and this script re-downloads and checksum-verifies each PDF, then extracts
plain text next to it for the pipeline/eval scripts to consume.
"""

import csv
import hashlib
import sys
from pathlib import Path

import requests
from pypdf import PdfReader

MANIFEST = Path("data/corpus_manifest.csv")
OUTPUT_DIR = Path("data/corpus")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def fetch_one(row: dict[str, str]) -> None:
    doc_id = row["doc_id"]
    pdf_path = OUTPUT_DIR / f"{doc_id}.pdf"

    if pdf_path.exists() and _sha256(pdf_path) == row["sha256"]:
        print(f"[skip] {doc_id}: already downloaded, checksum OK")
    else:
        print(f"[fetch] {doc_id}: {row['url']}")
        response = requests.get(
            row["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=60
        )
        response.raise_for_status()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(response.content)

        actual = _sha256(pdf_path)
        if actual != row["sha256"]:
            pdf_path.unlink()
            raise ValueError(
                f"checksum mismatch — expected {row['sha256']}, got {actual}. "
                "The source document may have changed; update data/corpus_manifest.csv."
            )

    txt_path = OUTPUT_DIR / f"{doc_id}.txt"
    txt_path.write_text(_extract_text(pdf_path), encoding="utf-8")
    print(f"[ok] {doc_id}: {row['pages']} pages -> {txt_path}")


def main() -> None:
    with MANIFEST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    failures: list[str] = []
    for row in rows:
        try:
            fetch_one(row)
        except Exception as exc:
            print(f"[FAIL] {row['doc_id']}: {exc}", file=sys.stderr)
            failures.append(row["doc_id"])

    if failures:
        print(f"\n{len(failures)}/{len(rows)} documents failed: {failures}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{len(rows)} documents fetched and verified.")


if __name__ == "__main__":
    main()
