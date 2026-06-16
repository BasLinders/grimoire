"""Corpus ingestion: extract clean text from web URLs and local documents.

Supported sources
-----------------
- **Web URL** — fetches HTML and strips boilerplate (scripts, nav, footer).
  Note: JavaScript-rendered pages are not supported; the raw HTML is parsed.
- **PDF** (.pdf) — extracts prose text and tables page-by-page via
  ``pdfplumber``.  Tables are rendered as Markdown pipe tables so stat
  blocks and ability-score grids survive extraction intact.
- **Word document** (.docx) — extracts paragraph text via ``python-docx``.
- **Markdown** (.md, .markdown) — strips syntax markers, returns plain text.
- **Excel workbook** (.xlsx) — each sheet is rendered as a Markdown pipe
  table.  Requires ``openpyxl``.
- **Plain text** (.txt) — returned as-is.
- **Image** (.png, .jpg, .jpeg, .tiff, .bmp, .gif) — OCR via
  ``pytesseract`` + ``Pillow``.  Requires both libraries **and** a system
  Tesseract install.  Raises ``ImportError`` with a helpful message if
  either is missing.

Typical usage
-------------
Write a single source to a directory::

    from grimoire_ai.corpus.ingest import ingest

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
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Cleaning levels
# ---------------------------------------------------------------------------

class CleaningLevel(str, Enum):
    """Controls how aggressively extracted text is post-processed.

    Attributes:
        MINIMAL:  Unicode normalisation and edge whitespace only.  Use when
            you want to preserve structure — poetry, stat blocks, tables.
        STANDARD: MINIMAL plus collapsing of multiple blank lines and
            runs of spaces/tabs.  Good default for most prose sources.
        THOROUGH: STANDARD plus removal of very short lines (likely
            navigation remnants, page numbers, or headers) and deduplication
            of consecutive identical paragraphs.  Best for web-scraped HTML
            where boilerplate stripping leaves residual fragments.
    """

    MINIMAL  = "minimal"
    STANDARD = "standard"
    THOROUGH = "thorough"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
_DOC_EXTS   = {".pdf", ".docx", ".xlsx", ".md", ".markdown", ".txt"} | _IMAGE_EXTS

# Thorough cleaning: lines with fewer than this many words are dropped.
_MIN_WORDS_THOROUGH = 4


def _clean(text: str, level: CleaningLevel = CleaningLevel.STANDARD) -> str:
    """Post-process extracted text according to ``level``.

    Args:
        text: Raw extracted text string.
        level: One of ``CleaningLevel.MINIMAL``, ``STANDARD``, ``THOROUGH``.

    Returns:
        Cleaned text string.
    """
    # Always normalise Unicode and strip edges — applies at every level.
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()

    if level == CleaningLevel.MINIMAL:
        return text

    # Standard and above: collapse whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    if level == CleaningLevel.THOROUGH:
        # Drop lines that are too short to be meaningful prose.
        lines = text.splitlines()
        lines = [
            line for line in lines
            if len(line.split()) >= _MIN_WORDS_THOROUGH or not line.strip()
        ]
        text = "\n".join(lines)

        # Deduplicate consecutive identical paragraphs.
        paragraphs = text.split("\n\n")
        deduped: list[str] = []
        prev = None
        for para in paragraphs:
            stripped = para.strip()
            if stripped != prev:
                deduped.append(para)
                prev = stripped
        text = "\n\n".join(deduped)

        # Re-collapse any blank lines introduced by the line filter.
        text = re.sub(r"\n{3,}", "\n\n", text)

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

def _clean_markdown_text(text: str, cleaning: CleaningLevel) -> str:
    """Apply markdown syntax stripping to a raw string (no file I/O)."""
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^```[^\n]*\n(.*?)^```", r"\1", text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    return _clean(text, level=cleaning)


def from_url(
    url: str,
    timeout: int = 15,
    cleaning: CleaningLevel = CleaningLevel.STANDARD,
) -> str:
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

    # If the URL path ends in .md/.markdown, or the server says text/plain,
    # treat the body as raw Markdown rather than HTML.
    content_type = resp.headers.get("Content-Type", "")
    url_path = urlparse(url).path.lower()
    is_markdown = url_path.endswith((".md", ".markdown")) or (
        "text/plain" in content_type and not "html" in content_type
    )
    if is_markdown:
        return _clean_markdown_text(resp.text, cleaning)

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise tags in-place.
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Prefer <main> or <article> for the main content block.
    body = soup.find("main") or soup.find("article") or soup.find("body") or soup
    return _clean(body.get_text(separator="\n"), level=cleaning)


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Convert a pdfplumber table (list-of-rows) to a Markdown pipe table.

    Empty cells are replaced with a single space so the pipe structure is
    preserved.  A separator row is inserted after the first (header) row.
    """
    def _cell(v: str | None) -> str:
        return (v or "").replace("\n", " ").strip() or " "

    rows = [[_cell(c) for c in row] for row in table if any(c for c in row)]
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)
    # Pad short rows
    rows = [r + [" "] * (col_count - len(r)) for r in rows]
    header = "| " + " | ".join(rows[0]) + " |"
    sep    = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body   = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
    return "\n".join(filter(None, [header, sep, body]))


def from_pdf(path: str, cleaning: CleaningLevel = CleaningLevel.STANDARD) -> str:
    """Extract text and tables from a PDF file.

    Uses ``pdfplumber`` to detect table regions on each page and render them
    as Markdown pipe tables.  Prose text outside those regions is extracted
    separately and interleaved with the tables in page order.  This preserves
    D&D stat blocks, ability-score grids, and other structured content that
    plain text extraction would garble.

    Args:
        path: Path to the ``.pdf`` file.
        cleaning: Cleaning level applied to the prose sections.

    Returns:
        Cleaned text with tables rendered as Markdown.

    Raises:
        ImportError: If ``pdfplumber`` is not installed.
        FileNotFoundError: If ``path`` does not exist.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required for PDF ingestion. "
            "Install it with: pip install pdfplumber"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    page_chunks: list[str] = []

    with pdfplumber.open(str(p)) as pdf:
        for page in pdf.pages:
            tables      = page.find_tables()
            table_bboxes = [t.bbox for t in tables]

            # Crop out table regions and extract surrounding prose.
            prose_page = page
            for bbox in table_bboxes:
                # pdfplumber.Page.filter() removes objects inside the bbox.
                prose_page = prose_page.filter(
                    lambda obj, bb=bbox: not (
                        bb[0] <= obj.get("x0", 0) <= bb[2]
                        and bb[1] <= obj.get("top", 0) <= bb[3]
                    )
                )
            prose_text = prose_page.extract_text() or ""

            # Render tables as Markdown.
            md_tables = [
                _table_to_markdown(t.extract())
                for t in tables
                if t.extract()
            ]

            # Interleave: prose first, then tables (pdfplumber tables are
            # ordered top-to-bottom so this approximates reading order).
            chunks = [prose_text] + md_tables
            page_chunks.append("\n\n".join(c for c in chunks if c.strip()))

    raw = "\n\n".join(page_chunks)
    return _clean(raw, level=cleaning)


def from_docx(path: str, cleaning: CleaningLevel = CleaningLevel.STANDARD) -> str:
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
    return _clean("\n\n".join(paragraphs), level=cleaning)


def from_xlsx(path: str, cleaning: CleaningLevel = CleaningLevel.STANDARD) -> str:
    """Extract text from an Excel workbook (.xlsx).

    Each worksheet is rendered as a Markdown pipe table using the same
    ``_table_to_markdown`` helper as the PDF extractor.  Sheets are separated
    by a heading line so the model can tell them apart.

    Args:
        path: Path to the ``.xlsx`` file.
        cleaning: Cleaning level applied to the combined output.

    Returns:
        All sheets rendered as Markdown tables, joined by newlines.

    Raises:
        ImportError: If ``openpyxl`` is not installed.
        FileNotFoundError: If ``path`` does not exist.
    """
    try:
        import openpyxl
    except ImportError as exc:
        raise ImportError(
            "openpyxl is required for XLSX ingestion. "
            "Install it with: pip install openpyxl"
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"XLSX file not found: {path}")

    wb = openpyxl.load_workbook(str(p), read_only=True, data_only=True)
    sections: list[str] = []
    for sheet in wb.worksheets:
        rows = [[str(cell.value) if cell.value is not None else "" for cell in row]
                for row in sheet.iter_rows()]
        rows = [row for row in rows if any(c.strip() for c in row)]
        if not rows:
            continue
        md = _table_to_markdown(rows)
        if md:
            sections.append(f"## {sheet.title}\n\n{md}")
    wb.close()

    raw = "\n\n".join(sections)
    return _clean(raw, level=cleaning)


def from_markdown(path: str, cleaning: CleaningLevel = CleaningLevel.STANDARD) -> str:
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

    # Use errors="replace" to handle non-UTF-8 Markdown files gracefully.
    text = p.read_text(encoding="utf-8", errors="replace")
    return _clean_markdown_text(text, cleaning)


def from_image(path: str, cleaning: CleaningLevel = CleaningLevel.THOROUGH) -> str:
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

    # OCR output is always cleaned at THOROUGH level regardless of the
    # caller's preference — raw OCR tends to have many short noise lines.
    return _clean(text, level=CleaningLevel.THOROUGH)


def from_txt(path: str, cleaning: CleaningLevel = CleaningLevel.STANDARD) -> str:
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
    # Use errors="replace" to handle non-UTF-8 files gracefully instead of
    # raising UnicodeDecodeError when called from a single-file ingest path.
    return _clean(p.read_text(encoding="utf-8", errors="replace"), level=cleaning)


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def from_file(
    path: str,
    cleaning: CleaningLevel = CleaningLevel.STANDARD,
) -> str:
    """Extract text from a local file, auto-detecting format by extension.

    Args:
        path: Path to a file with a supported extension
            (``.txt``, ``.pdf``, ``.docx``, ``.xlsx``, ``.md``,
            ``.markdown``, ``.png``, ``.jpg``, ``.jpeg``, ``.tiff``,
            ``.bmp``, ``.gif``).
        cleaning: Cleaning level applied to the extracted text.  Images
            always use ``THOROUGH`` regardless of this setting.

    Returns:
        Extracted plain-text string.

    Raises:
        ValueError: If the file extension is not recognised.
        FileNotFoundError: If ``path`` does not exist.
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return from_pdf(path, cleaning=cleaning)
    if ext == ".docx":
        return from_docx(path, cleaning=cleaning)
    if ext == ".xlsx":
        return from_xlsx(path, cleaning=cleaning)
    if ext in {".md", ".markdown"}:
        return from_markdown(path, cleaning=cleaning)
    if ext == ".txt":
        return from_txt(path, cleaning=cleaning)
    if ext in _IMAGE_EXTS:
        return from_image(path)  # always THOROUGH
    raise ValueError(
        f"Unsupported file extension {ext!r}. "
        f"Supported: {', '.join(sorted(_DOC_EXTS))}"
    )


def from_directory(
    path: str,
    recursive: bool = False,
    cleaning: CleaningLevel = CleaningLevel.STANDARD,
    on_progress: Optional[Callable[[str], None]] = None,
) -> dict[str, str]:
    """Extract text from all supported files in a directory.

    Args:
        path: Path to the directory.
        recursive: If ``True``, descend into subdirectories.
        cleaning: Cleaning level applied to each extracted file.
        on_progress: Optional callback called with a status string after
            each file is processed.  Falls back to ``print`` when ``None``.

    Returns:
        A ``{file_path: text}`` dict for every file that was successfully
        extracted.  Files with unsupported extensions are silently skipped.
        Files that raise extraction errors are skipped with a warning.

    Raises:
        FileNotFoundError: If ``path`` does not exist or is not a directory.
    """
    d = Path(path)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    _emit = on_progress if on_progress is not None else print

    glob = "**/*" if recursive else "*"
    results: dict[str, str] = {}
    for entry in sorted(d.glob(glob)):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in _DOC_EXTS:
            continue
        try:
            results[str(entry)] = from_file(str(entry), cleaning=cleaning)
            _emit(f"  ✓ {entry}")
        except Exception as exc:  # noqa: BLE001
            _emit(f"  ✗ skipping {entry}: {exc}")

    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def ingest(
    source: str,
    output_dir: Optional[str] = None,
    recursive: bool = False,
    timeout: int = 15,
    cleaning: CleaningLevel = CleaningLevel.STANDARD,
    on_progress: Optional[Callable[[str], None]] = None,
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

    _emit = on_progress if on_progress is not None else print

    out = Path(output_dir) if output_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    if is_url:
        text = from_url(source, timeout=timeout, cleaning=cleaning)
        if out is not None:
            dest = out / (_url_to_filename(source) + ".txt")
            dest.write_text(text, encoding="utf-8")
            _emit(f"  → {dest}")
            return None
        return text

    if is_dir:
        results = from_directory(
            source,
            recursive=recursive,
            cleaning=cleaning,
            on_progress=on_progress,
        )
        for file_path, text in results.items():
            src = Path(file_path)
            stem = src.stem
            dest = out / (stem + ".txt")
            # When two source files share the same stem (e.g. rules.pdf and
            # rules.docx), append the original extension to avoid silent overwrite.
            if dest.exists():
                stem = stem + "_" + src.suffix.lstrip(".")
                dest = out / (stem + ".txt")
            dest.write_text(text, encoding="utf-8")
        return None

    # Single file.
    text = from_file(source, cleaning=cleaning)
    if out is not None:
        dest = out / (source_path.stem + ".txt")
        dest.write_text(text, encoding="utf-8")
        _emit(f"  → {dest}")
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
    p.add_argument(
        "--cleaning",
        choices=[l.value for l in CleaningLevel],
        default=CleaningLevel.STANDARD.value,
        help=(
            "Text cleaning level. "
            "minimal=whitespace only; "
            "standard=collapse blank lines + spaces (default); "
            "thorough=standard + drop short lines + dedup paragraphs."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    result = ingest(
        source=args.source,
        output_dir=args.output,
        recursive=args.recursive,
        timeout=args.timeout,
        cleaning=CleaningLevel(args.cleaning),
    )
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
