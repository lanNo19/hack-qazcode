"""
parse_icd_descriptions.py

Builds a dictionary mapping ICD-10 codes (with dots) to their Russian text
descriptions by parsing the МКБ-10 section of each protocol in the JSONL corpus.

Output: icd_descriptions.json
  {
    "I50.0": "Застойная сердечная недостаточность",
    "I50.1": "Левожелудочковая недостаточность",
    ...
  }

Only codes with a dot are included (e.g. J67.6, I21.0) — bare parent codes
like I50 or J67 are skipped as they are less precise.

Usage:
    python parse_icd_descriptions.py
    python parse_icd_descriptions.py --corpus path/to/protocols_corpus.jsonl
    python parse_icd_descriptions.py --output path/to/icd_descriptions.json
"""

import json
import re
import argparse
from collections import defaultdict


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# ICD-10 code with dot: one capital letter + 2 digits + dot + 1-2 digits
# Examples: I50.0, J67.6, I21.00, R57.0, Z03.8
ICD_RE = re.compile(r'\b([A-Z]\d{2}\.\d{1,2})\b')

# МКБ-10 section marker — appears near the start of each protocol
MKB_MARKER_RE = re.compile(r'МКБ-10')

# Stop boundary after the МКБ-10 section: a line that starts with a digit
# (next numbered section) or a Roman numeral section header, or common
# boilerplate markers. We search for this after the last ICD code match.
SECTION_BREAK_RE = re.compile(
    r'\n[ \t]*(?:\d+[\. ]|[IVXЛC]+[\. ]|Сокращения|Дата разработки|Пользователи|Категория)'
)


def clean_description(raw: str) -> str:
    """
    Strip the separator characters between a code and its description,
    then collapse whitespace.

    The separator is typically ' – ' (em dash) or ' - ' (hyphen), sometimes
    with extra spaces or newlines due to PDF line wrapping.
    """
    # Remove leading separator (–, —, -, :, whitespace)
    desc = re.sub(r'^[\s\u2013\u2014\-:]+', '', raw)
    # Collapse internal whitespace (handles line-wrapped descriptions)
    desc = re.sub(r'\s+', ' ', desc).strip()
    return desc


def extract_icd_section(text: str) -> str | None:
    """
    Locate the МКБ-10 section in the protocol text and return the substring
    from the МКБ-10 marker to the end of the ICD code list.

    Returns None if no МКБ-10 marker is found.
    """
    marker = MKB_MARKER_RE.search(text)
    if not marker:
        return None

    # Take a generous window after the marker — the ICD list is usually
    # 50–300 lines long, rarely more than 5000 characters
    window_start = marker.start()
    window_end = min(window_start + 6000, len(text))
    return text[window_start:window_end]


def parse_icd_descriptions(text: str) -> dict[str, str]:
    """
    Parse the МКБ-10 section of one protocol text.

    Returns a dict of {icd_code_with_dot: description}.
    """
    section = extract_icd_section(text)
    if not section:
        return {}

    matches = list(ICD_RE.finditer(section))
    if not matches:
        return {}

    result = {}

    for i, m in enumerate(matches):
        code = m.group(1)

        # Text starts right after the code
        desc_start = m.end()

        # Text ends at the start of the next code
        if i + 1 < len(matches):
            desc_end = matches[i + 1].start()
        else:
            # Last code: end at the next section break or end of window
            stop = SECTION_BREAK_RE.search(section, desc_start)
            # Also stop at the first bare number on its own line (e.g. "4.")
            # which signals the next numbered section
            bare_section = re.search(r'\n\d+\.', section[desc_start:])
            candidates = []
            if stop:
                candidates.append(stop.start())
            if bare_section:
                candidates.append(desc_start + bare_section.start())
            desc_end = min(candidates) if candidates else len(section)

        raw = section[desc_start:desc_end]
        desc = clean_description(raw)

        if desc and len(desc) > 2:  # skip empty or garbage descriptions
            # If we already have this code from another protocol, keep the
            # longer/more descriptive version
            if code not in result or len(desc) > len(result[code]):
                result[code] = desc

    return result


def build_icd_dict(jsonl_path: str) -> dict[str, str]:
    """
    Read the full JSONL corpus and build the combined ICD-10 → description dict.
    Later entries overwrite earlier ones only if the description is longer.
    """
    combined: dict[str, str] = {}
    protocol_count = 0
    code_count = 0
    conflict_count = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = data.get('text', '')
            protocol_count += 1

            descriptions = parse_icd_descriptions(text)
            for code, desc in descriptions.items():
                if code not in combined:
                    combined[code] = desc
                    code_count += 1
                elif len(desc) > len(combined[code]):
                    combined[code] = desc  # keep longer description
                    conflict_count += 1

    print(f"Protocols scanned : {protocol_count}")
    print(f"Unique codes found : {len(combined)}")
    print(f"Conflicts resolved : {conflict_count} (kept longer description)")
    return combined


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def debug_protocol_icd(text: str, source_file: str = '?'):
    """Print what ICD codes and descriptions were extracted from one protocol."""
    descriptions = parse_icd_descriptions(text)
    print(f"\n{'='*60}")
    print(f"Protocol: {source_file}")
    print(f"Codes extracted: {len(descriptions)}")
    for code, desc in list(descriptions.items())[:20]:
        print(f"  {code:10s}: {desc}")
    if len(descriptions) > 20:
        print(f"  ... and {len(descriptions) - 20} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build ICD-10 code → description dict from JSONL corpus')
    parser.add_argument('--corpus', default='corpus/protocols_corpus.jsonl',
                        help='Path to protocols_corpus.jsonl')
    parser.add_argument('--output', default='icd_descriptions.json',
                        help='Output JSON file path')
    args = parser.parse_args()

    print(f"Reading corpus: {args.corpus}")
    icd_dict = build_icd_dict(args.corpus)

    # Sort by code for readability
    icd_dict_sorted = dict(sorted(icd_dict.items()))

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(icd_dict_sorted, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(icd_dict_sorted)} codes → {args.output}")

    # Print a sample
    print("\nSample entries:")
    for code, desc in list(icd_dict_sorted.items())[:15]:
        print(f"  {code:10s}: {desc}")