"""Corpus ingestion: extract clean text from web URLs and local documents.

Supported sources
-----------------
- **Web URL** — fetches HTML and strips boilerplate (scripts, nav, footer).
  Note: JavaScript-rendered pages are not supported; the raw HTML is parsed.
- **PDF** (.pdf) — extracts text page-by-page via ``pypdf``.
- **Word document** (.docx) — extracts paragraph text via ``python-docx``.
- **Markdown** (.md, .markdown) — strips syntax markers, returns plain text.
- **Plain text** (.txt) — returned as-is.
- **Image** (.png, .jpg, .jpeg, .tiff, .bmp, .gif) — OCR via
  ``pytesseract`` + ``Pillow``.  Requires both libraries **and** a system
  Tesseract install.  Raises ``ImportError`` with a helpful message if
  either is missing.

Typical usage
-------------
Write a single source to a directory::

    from grimoire.corpus.ingest import ingest

    ingest("https://www.dndbeyond.com/sources/basic-rules", output_dir="data/raw/")
    ingest("docs/phb_excerpt.pdf",   output_dir="data/raw/")
    ingest("notes/",                 output_dir="data/raw/", recursive=True)

Or get the text directly (no file written)::

    text = ingest("docs/phb_excerpt.pdf")

CLI::

    python -m grimoire.corpus.ingest --source URL_OR_PATH --output data/raw/
    python -m grimoire.corpus.ingest --source docs/ --output data/raw/ --recursive
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
_DOC_EXTS   = {".pdf", ".docx", ".md", ".markdown", ".txt"} | _IMAGE_EXTS


def _clean(text: str) -> str:
    """Normalise Unicode, collapse whitespace, strip leading/trailing blanks."""
    text = unicodedata.normalize("NFKC", text)
    # Collapse runs of blank lines to a single blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs within a line.
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _url_to_filename(url: str) -> str:
    """Derive a safe filename stem from a URL."""
    parsed = urlparse(url)
    # Use the path component; fall back to the hostname.
    slug = parsed.path.strip("/").replace("/", "_") or parsed.hostname or "page"
    # Remove characters that are unsafe in filenames.
    slug = re.sub(r"[^\w\-.]", "_", slug)
    return slug[:128] or "page"


# ---------------------------------------------------------------------------
# Per-format extractors
# ---------------------------------------------------------------------------

def from_url(url: str, timeout: int = 15) -> str:
    """Fetch a web page and return its main text content.

    Strips ``<script>``, ``<style>``, ``<nav>``, ``<footer>``, ``<header>``,
    and ``<aside>`` tags.  Prefers ``<main>`` or ``<article>`` over the full
    ``<body>`` when they are present.

    Args:
        url: Fully-qualified HTTP/HTTPS URL.
        timeout: Request timeout in seconds.

    Returns:
        Cleaned plain-text string.

    Raises:
        ImportError: If ``beautifulsoup4`` is not installed.
        RuntimeError: If the HTTP request fails or returns a non-200 status.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "beautifulsoup4 is required for URL ingestion. "
            "Install it with: pip install beautifulsoup4"
        ) from exc

    import requests  # always available (listed in core deps)

    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "GrimoireBot/1.0"})
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch {url!r}: {exc}") from exc

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise tags in-place.
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Prefer <main> or <article> for the main content block.
    body = soup.find("main") or soup.find("article") or soup.find("body") or soup
    return _clean(body.get_text(separator="\n"))


def from_pdf(path: str) -> str:
    """Extract text from a PDF file.

    Args:
        path: Path to the ``.pdf`` file.

    Returns:
        Concatenated plain-text from all pages.

    Raises:
        ImportError: If ``pypdf`` is not installed.
        FileNotFoundError: If ``path`` does not exist.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF ingestion. "
            "Install it with: pip install pypdf"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(p))
    pages = [page.extract_text() or "" for page in reader.pages]
    return _clean("\n\n".join(pages))


def from_docx(path: str) -> str:
    """Extract text from a Word document (.docx).

    Args:
        path: Path to the ``.docx`` file.

    Returns:
        Paragraph text joined by newlines.

    Raises:
        ImportError: If ``python-docx`` is not installed.
        FileNotFoundError: If ``path`` does not exist.
    """
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for DOCX ingestion. "
            "Install it with: pip install python-docx"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DOCX not found: {path}")

    doc = Document(str(p))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return _clean("\n\n".join(paragraphs))


def from_markdown(path: str) -> str:
    """Extract plain text from a Markdown file by stripping syntax markers.

    Removes: headings (``#``), bold/italic (``*``, ``_``), inline code
    (`` ` ``), links (``[text](url)``), images (``![alt](url)``), horizontal
    rules, and blockquote markers.

    Args:
        path: Path to the ``.md`` or ``.markdown`` file.

    Returns:
        Plain-text string with Markdown syntax removed.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")

    text = p.read_text(encoding="utf-8")
    # Images before links (same pattern, different prefix).
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Headings.
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bold / italic.
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
    # Inline code.
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Fenced code blocks — keep the content, remove the fences.
    text = re.sub(r"^```[^\n]*\n(.*?)^```", r"\1", text, flags=re.MULTILINE | re.DOTALL)
    # Horizontal rules.
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Blockquote markers.
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    return _clean(text)


def from_image(path: str) -> str:
    """Extract text from an image file via OCR (Tesseract).

    Requires ``pytesseract``, ``Pillow``, **and** a system Tesseract install.
    On Windows: https://github.com/UB-Mannheim/tesseract/wiki
    On Linux: ``apt install tesseract-ocr``

    Args:
        path: Path to the image file (.png, .jpg, .jpeg, .tiff, .bmp, .gif).

    Returns:
        OCR-extracted text string.

    Raises:
        ImportError: If ``pytesseract`` or ``Pillow`` is not installed.
        FileNotFoundError: If ``path`` does not exist.
        RuntimeError: If Tesseract is not found on the system PATH.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "pytesseract and Pillow are required for image OCR. "
            "Install with: pip install pytesseract Pillow\n"
            "You also need Tesseract on your system: "
            "https://github.com/UB-Mannheim/tesseract/wiki"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        text = pytesseract.image_to_string(Image.open(str(p)))
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract not found. Install it:\n"
            "  Linux:   apt install tesseract-ocr\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
        ) from exc

    return _clean(text)


def from_txt(path: str) -> str:
    """Read a plain-text file.

    Args:
        path: Path to the ``.txt`` file.

    Returns:
        Cleaned text content.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    return _clean(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def from_file(path: str) -> str:
    """Extract text from a local file, auto-detecting format by extension.

    Args:
        path: Path to a file with a supported extension
            (``.txt``, ``.pdf``, ``.docx``, ``.md``, ``.markdown``,
            ``.png``, ``.jpg``, ``.jpeg``, ``.tiff``, ``.bmp``, ``.gif``).

    Returns:
        Extracted plain-text string.

    Raises:
        ValueError: If the file extension is not recognised.
        FileNotFoundError: If ``path`` does not exist.
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return from_pdf(path)
    if ext == ".docx":
        return from_docx(path)
    if ext in {".md", ".markdown"}:
        return from_markdown(path)
    if ext == ".txt":
        return from_txt(path)
    if ext in _IMAGE_EXTS:
        return from_image(path)
    raise ValueError(
        f"Unsupported file extension {ext!r}. "
        f"Supported: {', '.join(sorted(_DOC_EXTS))}"
    )


def from_directory(
    path: str,
    recursive: bool = False,
) -> dict[str, str]:
    """Extract text from all supported files in a directory.

    Args:
        path: Path to the directory.
        recursive: If ``True``, descend into subdirectories.

    Returns:
        A ``{file_path: text}`` dict for every file that was successfully
        extracted.  Files with unsupported extensions are silently skipped.
        Files that raise extraction errors are skipped with a warning printed
        to stderr.

    Raises:
        FileNotFoundError: If ``path`` does not exist or is not a directory.
    """
    d = Path(path)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    glob = "**/*" if recursive else "*"
    results: dict[str, str] = {}
    for entry in sorted(d.glob(glob)):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in _DOC_EXTS:
            continue
        try:
            results[str(entry)] = from_file(str(entry))
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: skipping {entry}: {exc}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest(
    source: str,
    output_dir: Optional[str] = None,
    recursive: bool = False,
    timeout: int = 15,
) -> Optional[str]:
    """Ingest a source and optionally write the result to ``output_dir``.

    ``source`` may be:
    - An HTTP/HTTPS URL
    - A path to a supported file
    - A path to a directory (all supported files are processed)

    When ``output_dir`` is given each result is written as a ``.txt`` file
    whose name is derived from the source.  When ``output_dir`` is ``None``
    the text is returned directly (only valid for single-file and URL sources;
    directory ingestion always requires ``output_dir``).

    Args:
        source: URL, file path, or directory path.
        output_dir: Directory to write ``.txt`` output files.  Created if it
            does not exist.
        recursive: When ``source`` is a directory, descend into
            subdirectories.
        timeout: HTTP request timeout in seconds (URL sources only).

    Returns:
        The extracted text when ``output_dir`` is ``None`` and ``source`` is
        a single file or URL.  ``None`` when writing to disk.

    Raises:
        ValueError: If ``source`` is a directory and ``output_dir`` is
            ``None``.
    """
    source_path = Path(source)
    is_url = source.startswith("http://") or source.startswith("https://")
    is_dir = source_path.is_dir()

    if is_dir and output_dir is None:
        raise ValueError("output_dir is required when source is a directory.")

    out = Path(output_dir) if output_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    if is_url:
        text = from_url(source, timeout=timeout)
        if out is not None:
            dest = out / (_url_to_filename(source) + ".txt")
            dest.write_text(text, encoding="utf-8")
            print(f"  → {dest}")
            return None
        return text

    if is_dir:
        results = from_directory(source, recursive=recursive)
        for file_path, text in results.items():
            stem = Path(file_path).stem
            dest = out / (stem + ".txt")
            dest.write_text(text, encoding="utf-8")
            print(f"  → {dest}")
        return None

    # Single file.
    text = from_file(source)
    if out is not None:
        dest = out / (source_path.stem + ".txt")
        dest.write_text(text, encoding="utf-8")
        print(f"  → {dest}")
        return None
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest text from URLs, PDFs, DOCX, Markdown, or images into corpus .txt files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", required=True,
                   help="URL, file path, or directory path to ingest.")
    p.add_argument("--output", default=None,
                   help="Output directory for .txt files. Prints to stdout if omitted.")
    p.add_argument("--recursive", action="store_true",
                   help="Recurse into subdirectories (only applies when --source is a directory).")
    p.add_argument("--timeout", type=int, default=15,
                   help="HTTP request timeout in seconds.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    result = ingest(
        source=args.source,
        output_dir=args.output,
        recursive=args.recursive,
        timeout=args.timeout,
    )
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
