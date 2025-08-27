"""
SEC Risk Factors / Market Risk Extractor (refactored, resilient, SGML-aware, year-filtered)

Pulls latest and previous SEC EDGAR filings for a ticker (10-K / 10-Q), extracts:
  - Item 1A. Risk Factors (preferred), or
  - Item 3. Quantitative and Qualitative Disclosures About Market Risk (fallback for 10-Q),
and returns:
    (current_text, previous_text, metadata_current, metadata_previous)

CLI:
    python data_collection/SEC_Edgar_Downloader.py \
        --ticker NVDA --form 10-Q --email dvschenone@ucsd.edu \
        --section auto --min-year 2019
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import List, Optional, Tuple, Pattern, Callable

from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

# ----------------------------
# Config
# ----------------------------
DEFAULT_DATA_DIR = Path("sec-edgar-filings")


@dataclass
class FilingMetadata:
    ticker: str
    company: str
    form: str
    date: str  # YYYY-MM-DD


# ----------------------------
# Helpers
# ----------------------------
def _normalize_form(form: str) -> str:
    f = form.upper().strip()
    return f.replace("10Q", "10-Q").replace("10K", "10-K")


def _parse_year_from_accession_dir(dirname: str) -> Optional[int]:
    """
    Accession dir examples: 0001012870-99-001954 -> 1999
                            0001045810-24-000067 -> 2024
    """
    m = re.search(r"-(\d{2})-", dirname)
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 70 else 2000 + yy  # pivot at 1970


# ----------------------------
# Downloading & Discovery
# ----------------------------
def ensure_download(
    ticker: str,
    form: str,
    email: str,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> None:
    """Download filings for ticker/form into data_dir (version-safe)."""
    data_dir.mkdir(parents=True, exist_ok=True)

    ctor_params = set(signature(Downloader).parameters.keys())
    if {"user_agent", "download_folder"} <= ctor_params:
        dl = Downloader(user_agent=email, download_folder=str(data_dir))
    elif {"company_email", "download_folder"} <= ctor_params:
        dl = Downloader(company_email=email, download_folder=str(data_dir))
    else:
        dl = Downloader(email, str(data_dir))

    dl.get(form, ticker)


def _score_doc_name(p: Path) -> int:
    """Heuristic to pick the 'main' HTML doc inside an accession. Higher is better."""
    name = p.name.lower()
    score = 0
    if "primary-document" in name:
        score += 200
    if "10-q" in name or "10k" in name or "10-k" in name or "form10" in name:
        score += 120
    if name.endswith(".htm") or name.endswith(".html"):
        score += 50
    if "index" in name or "cover" in name or "exhibit" in name:
        score -= 40
    try:
        score += min(int(p.stat().st_size // 1024), 500)  # up to +500 for big files
    except Exception:
        pass
    return score


def _best_doc_in_accession(dir_path: Path) -> Optional[Path]:
    """Prefer HTML/HTM; only use .txt if no HTML exists (and avoid full-submission if possible)."""
    files = [f for f in dir_path.iterdir() if f.is_file()]
    htmls = [f for f in files if f.suffix.lower() in {".html", ".htm"}]
    if htmls:
        htmls.sort(key=_score_doc_name, reverse=True)
        return htmls[0]

    txts = [f for f in files if f.suffix.lower() == ".txt"]
    if not txts:
        return None
    # Avoid full-submission.txt if there is any other large .txt
    txts.sort(
        key=lambda p: (
            0 if "full-submission" in p.name.lower() else 1,  # deprioritize full-submission
            p.stat().st_size if p.exists() else 0,
        ),
        reverse=True,
    )
    return txts[0]


def list_filing_files(
    ticker: str,
    form: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    min_year: int = 2019,
) -> List[Path]:
    """
    Return ONE best candidate file per accession directory, newest by accession YEAR (not mtime),
    filtered by min_year. Structure: <data_dir>/<ticker>/<form>/<accession>/files...
    """
    filings_root = data_dir / ticker / form
    if not filings_root.exists():
        return []

    rows: List[Tuple[int, Path]] = []
    for d in filings_root.iterdir():
        if not d.is_dir():
            continue
        year = _parse_year_from_accession_dir(d.name)
        if year is None or year < min_year:
            continue
        best = _best_doc_in_accession(d)
        if best:
            rows.append((year, best))

    # newest year first; if same year, fall back to file mtime
    rows.sort(key=lambda t: (t[0], int(t[1].stat().st_mtime)), reverse=True)
    return [p for _, p in rows]


# ----------------------------
# SGML-aware Parsing & Extraction
# ----------------------------
def _extract_primary_from_sgml(content: str) -> Optional[str]:
    """
    Parse EDGAR SGML full-submission.txt into sub-documents and return the best <TEXT> block.
    Preference order: TYPE=10-Q/10-K, then HTML-like, then largest TEXT block.
    Returns the raw TEXT (could be HTML or plain text).
    """
    s = content.replace("\r\n", "\n").replace("\r", "\n")
    docs = []
    for m in re.finditer(r"<DOCUMENT>(.*?)</DOCUMENT>", s, flags=re.DOTALL | re.IGNORECASE):
        block = m.group(1)
        typem = re.search(r"<TYPE>\s*([A-Za-z0-9\-]+)", block, flags=re.IGNORECASE)
        doc_type = typem.group(1).upper() if typem else ""
        textm = re.search(r"<TEXT>(.*?)</TEXT>", block, flags=re.DOTALL | re.IGNORECASE)
        if not textm:
            continue
        text = textm.group(1)
        is_html_like = ("<html" in text.lower()) or ("</table>" in text.lower()) or ("</div>" in text.lower())
        docs.append((doc_type, is_html_like, len(text), text))

    if not docs:
        return None

    def score(d):
        doc_type, is_html_like, size, _ = d
        type_score = 100 if doc_type in {"10-Q", "10K", "10-K"} else 0
        html_score = 10 if is_html_like else 0
        return (type_score, html_score, size)

    docs.sort(key=score, reverse=True)
    return docs[0][3]


def read_text_from_file(file_path: Path) -> str:
    """
    Read file and return plain text. Handles:
      - HTML/HTM -> strip tags
      - TXT SGML (full-submission) -> extract best sub-document from <TEXT>, then parse
      - TXT plain -> return as-is
    """
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    lower_head = content[:4096].lower()

    # SGML full-submission?
    if "<document>" in lower_head and "<text>" in lower_head:
        sub = _extract_primary_from_sgml(content)
        if sub:
            content = sub
            lower_head = content[:4096].lower()
            # print(f"ℹ️  Parsed SGML subdocument from {file_path.name}")

    # HTML or HTML-like?
    if ("<html" in lower_head) or ("</table>" in lower_head) or ("</div>" in lower_head):
        try:
            soup = BeautifulSoup(content, "lxml")
        except Exception:
            soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text(separator="\n")
    else:
        text = content

    # Normalize punctuation and whitespace for regex stability
    text = (
        text.replace("–", "-")
            .replace("—", "-")
            .replace("\xa0", " ")
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------- TOC handling helpers ----------
def _strip_table_of_contents(text: str) -> str:
    """
    Remove Table of Contents to avoid false-positive matches like 'Item 1A ..... 30'.
    We cut from 'TABLE OF CONTENTS' to the first 'PART I' or 'PART II'. Also drop dot-leader lines.
    """
    t = text
    m = re.search(r"table\s+of\s+contents", t, flags=re.IGNORECASE)
    if m:
        after = t[m.end():]
        mpart = re.search(r"\n\s*part\s+(i|ii)\b", after, flags=re.IGNORECASE)
        if mpart:
            t = t[:m.start()] + after[mpart.start():]
    lines = []
    for line in t.splitlines():
        if re.search(r"\.{2,}\s*\d{1,4}\s*$", line):
            continue
        lines.append(line)
    return "\n".join(lines)


def _heading_looks_like_toc(heading_line: str) -> bool:
    if re.search(r"\.{2,}\s*\d{1,4}\s*$", heading_line):
        return True
    if re.search(r"\s\d{1,4}\s*$", heading_line) and len(heading_line) < 120:
        return True
    return False


def _postprocess_section(text: str) -> Optional[str]:
    body = re.sub(r"\n\s+\n", "\n\n", text.strip())
    return body or None


def _first_non_toc_match(text: str, patterns: List[Pattern[str]]) -> Optional[str]:
    for pat in patterns:
        for m in pat.finditer(text):
            # Grab the heading line and skip TOC entries
            span_start = m.start()
            pre_start = text.rfind("\n", 0, span_start) + 1
            heading_line = text[pre_start: text.find("\n", span_start)]
            if _heading_looks_like_toc(heading_line):
                continue
            groups = [g for g in m.groups() if isinstance(g, str)]
            body = max((g.strip() for g in groups), key=len, default="")
            if body:
                return _postprocess_section(body)
    return None


# ---------- Item 1A (Risk Factors) patterns ----------
_ITEM_1A_RE: Pattern[str] = re.compile(
    r"""
    ^\s*item\s*1a\.?\s*[-–:]?\s*risk\s*factors\.?\s*$     # heading
    (.*?)                                                 # body
    ^\s*item\s*1b\.?\b|^\s*item\s*2\.?\b|^\s*part\s*ii\b  # next section
    """,
    re.IGNORECASE | re.DOTALL | re.MULTILINE | re.VERBOSE,
)
_ITEM_1A_ALT: List[Pattern[str]] = [
    re.compile(
        r"""
        ^\s*item\s*1a\.?\s*(?:[-–:]?\s*)?(?:risk\s*factors.*?)?\s*$  # flexible
        (.*?)                                                        # body
        ^\s*item\s*1b\.?\b|^\s*item\s*2\.?\b|^\s*part\s*ii\b
        """,
        re.IGNORECASE | re.DOTALL | re.MULTILINE | re.VERBOSE,
    ),
    re.compile(
        r"""
        ^\s*item\s*1a\b.*?$   # any 1A line
        (.*?)                 # body
        ^\s*item\s*1b\b|^\s*item\s*2\b|^\s*part\s*ii\b
        """,
        re.IGNORECASE | re.DOTALL | re.MULTILINE | re.VERBOSE,
    ),
    re.compile(
        r"item\s*1a.*?(risk\s*factors.*?)?(.*?)(?=item\s*1b\b|item\s*2\b|part\s*ii\b)",
        re.IGNORECASE | re.DOTALL,
    ),
]

# ---------- Item 3 (Market Risk) patterns ----------
_ITEM_3_RE: Pattern[str] = re.compile(
    r"""
    ^\s*item\s*3\.?\s*[-–:]?\s*quantitative\s+and\s+qualitative\s+disclosures\s+about\s+market\s+risk\.?\s*$  # heading
    (.*?)                                                                                                     # body
    ^\s*item\s*4\.?\b|^\s*part\s*ii\b|^\s*item\s*1\.?\b                                                       # next
    """,
    re.IGNORECASE | re.DOTALL | re.MULTILINE | re.VERBOSE,
)
_ITEM_3_ALT: List[Pattern[str]] = [
    re.compile(
        r"""
        ^\s*item\s*3\b.*?market\s+risk.*?$   # flexible heading containing 'market risk'
        (.*?)                                # body
        ^\s*item\s*4\.?\b|^\s*part\s*ii\b|^\s*item\s*1\.?\b
        """,
        re.IGNORECASE | re.DOTALL | re.MULTILINE | re.VERBOSE,
    ),
]


def _extract_with_patterns(text: str, pats: List[Pattern[str]]) -> Optional[str]:
    return _first_non_toc_match(text, pats)


def extract_item_1a(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = _strip_table_of_contents(text)
    body = _extract_with_patterns(cleaned, [_ITEM_1A_RE] + _ITEM_1A_ALT)
    if body:
        return body
    # Minimalist 10-Q language:
    nm = re.search(
        r"no\s+material\s+changes\s+to\s+our\s+risk\s+factors",
        cleaned,
        re.IGNORECASE,
    )
    if nm:
        return "No material changes to our risk factors since the most recent Form 10-K."
    return None


def extract_item_3(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = _strip_table_of_contents(text)
    return _extract_with_patterns(cleaned, [_ITEM_3_RE] + _ITEM_3_ALT)


def _safe_extract(path: Path, extractor: Callable[[str], Optional[str]]) -> Optional[str]:
    try:
        return extractor(read_text_from_file(path))
    except Exception as e:
        print(f"⚠️  Could not extract from {path}: {e}")
        return None


# ----------------------------
# Metadata helpers
# ----------------------------
def infer_company_from_path(p: Path) -> str:
    # .../sec-edgar-filings/<ticker>/<form>/<accession>/<file>
    return p.parts[-4] if len(p.parts) >= 4 else ""


def infer_date_from_file(p: Path) -> str:
    # Use modification time as proxy
    dt = datetime.fromtimestamp(p.stat().st_mtime)
    return dt.strftime("%Y-%m-%d")


def build_metadata(p: Path, ticker: str, form: str) -> FilingMetadata:
    return FilingMetadata(
        ticker=ticker,
        company=infer_company_from_path(p),
        form=form,
        date=infer_date_from_file(p),
    )


# ----------------------------
# Public API
# ----------------------------
def _choose_section_text(
    file_path: Path,
    form: str,
    section_preference: str = "auto",  # "auto" | "risk" | "item3"
) -> Tuple[Optional[str], str]:
    """
    Returns (text, section_used): "Item 1A", "Item 3", or "None".
    """
    pref = section_preference.lower()

    if pref == "risk":
        t = _safe_extract(file_path, extract_item_1a)
        return (t, "Item 1A" if t else "None")

    if pref == "item3":
        t = _safe_extract(file_path, extract_item_3)
        return (t, "Item 3" if t else "None")

    # auto:
    if form == "10-K":
        t = _safe_extract(file_path, extract_item_1a)
        return (t, "Item 1A" if t else "None")

    # 10-Q: try Item 1A then fallback Item 3
    t = _safe_extract(file_path, extract_item_1a)
    if t:
        return (t, "Item 1A")
    t = _safe_extract(file_path, extract_item_3)
    return (t, "Item 3" if t else "None")


def prepare_compare_payload(
    ticker: str,
    form: str = "10-Q",
    email: str = "example@example.com",
    data_dir: Path = DEFAULT_DATA_DIR,
    section_preference: str = "auto",  # "auto" | "risk" | "item3"
    min_year: int = 2019,
) -> Tuple[Optional[str], str, FilingMetadata, FilingMetadata]:
    """
    Download filings and return:
        (current_text, previous_text, metadata_current, metadata_previous)
    """
    form = _normalize_form(form)
    ensure_download(ticker, form, email=email, data_dir=data_dir)

    files = list_filing_files(ticker, form, data_dir=data_dir, min_year=min_year)
    if not files:
        raise FileNotFoundError(
            f"No filings found under {data_dir}/{ticker}/{form} at or after {min_year}"
        )

    latest_file = files[0]
    prev_file = files[1] if len(files) > 1 else files[0]

    print(f"ℹ️  Latest file:   {latest_file}")
    print(f"ℹ️  Previous file: {prev_file}")

    current_text, current_used = _choose_section_text(latest_file, form, section_preference)
    previous_text, previous_used = _choose_section_text(prev_file, form, section_preference)

    print(f"🔎 Section used (latest):   {current_used}")
    print(f"🔎 Section used (previous): {previous_used}")

    meta_latest = build_metadata(latest_file, ticker, form)
    meta_prev = build_metadata(prev_file, ticker, form)

    return current_text if current_text else None, previous_text or "", meta_latest, meta_prev


# ----------------------------
# CLI
# ----------------------------
def _main() -> None:
    ap = argparse.ArgumentParser(description="Extract Item 1A / Item 3 from latest SEC filings")
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--form", default="10-Q")
    ap.add_argument("--email", required=True, help="Contact email for SEC requests")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--section", default="auto", choices=["auto", "risk", "item3"],
                    help="auto = 10-K: Item 1A; 10-Q: Item 1A then fallback Item 3")
    ap.add_argument("--min-year", type=int, default=2019,
                    help="Ignore accessions older than this year based on directory name")
    args = ap.parse_args()

    print(f"⬇️  Downloading {args.form} filings for {args.ticker} into {args.data_dir} ...")
    current_text, previous_text, meta_curr, meta_prev = prepare_compare_payload(
        ticker=args.ticker,
        form=args.form,
        email=args.email,
        data_dir=Path(args.data_dir),
        section_preference=args.section,
        min_year=args.min_year,
    )
    print("✅ Download + parse complete.\n")

    def preview(s: Optional[str]) -> str:
        if s is None:
            return "<None>"
        s = s.strip().replace("\n", " ")
        return (s[:400] + "…") if len(s) > 400 else s

    print("CURRENT METADATA:", meta_curr)
    print("PREVIOUS METADATA:", meta_prev)
    print("CURRENT SECTION PREVIEW:", preview(current_text))
    print("PREVIOUS SECTION PREVIEW:", preview(previous_text))


if __name__ == "__main__":
    _main()
