"""Extract event and field descriptions from Spotify's Technical Log ReadMe PDF.

    python scripts/extract_techlog_docs.py [--pdf FILE] [--out FILE]

Writes techlog_docs.json, which build_database.py uses to describe every event
type in silver.log_event_types and every column of the techlog views. Run it
again only when a newer export comes with a newer ReadMe. Needs pypdf.

The context_* fields and timestamp_utc are left out: every event has them and
the build stores them as the columns of silver.log_events.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = ROOT / "data" / "raw" / "Spotify Technical Log Information" / "ReadMeFirst_TechnicalLogInformation.pdf"
DEFAULT_OUT = ROOT / "data" / "external" / "techlog_docs.json"

# In layout mode each event starts with its name centred on a line of its own,
# then "Category: ...", a description, and a two-column table of fields: the
# field name at the left margin, its description (often wrapped) to the right.
# Blank lines separate table rows.
EVENT_NAME = re.compile(r"^\s{10,}([A-Za-z][A-Za-z0-9_]*)$")
CATEGORY = re.compile(r"^\s*Category:\s*(\S+)$")
TABLE_HEADER = re.compile(r"^\s*item name\s+item description$")
FIELD_ROW = re.compile(r"^\s{0,3}([A-Za-z][A-Za-z0-9_]*)(?:\s{2,}(.*))?$")


def page_lines(reader: PdfReader) -> list[str | None]:
    lines: list[str | None] = []
    for page in reader.pages:
        lines.extend(line.rstrip() for line in page.extract_text(extraction_mode="layout").splitlines())
        lines.append(None)  # page break: a table row never continues past it without a field name
    return lines


def category_after(lines: list[str | None], i: int) -> str | None:
    for line in lines[i + 1 :]:
        if line:
            match = CATEGORY.match(line)
            return match.group(1) if match else None
    return None


def parse(lines: list[str | None]) -> dict:
    events: dict = {}
    event = None
    in_table = False
    row_name, row_text = None, []

    def end_row():
        nonlocal row_name, row_text
        text = " ".join(" ".join(row_text).split())
        if event is not None and in_table:
            if row_name:
                event["fields"].setdefault(row_name, text)
            elif text and event["fields"]:
                # The description of the last row carried over to a new page.
                last = next(reversed(event["fields"]))
                event["fields"][last] = f"{event['fields'][last]} {text}".strip()
        row_name, row_text = None, []

    for i, line in enumerate(lines):
        if line is None or not line.strip():
            end_row()
            continue
        name = EVENT_NAME.match(line)
        if name and (category := category_after(lines, i)):
            end_row()
            event = events.setdefault(name.group(1), {"category": category, "description": "", "fields": {}})
            in_table = False
            continue
        if event is None:
            continue
        if not in_table:
            if TABLE_HEADER.match(line):
                in_table = True
            elif not CATEGORY.match(line):
                event["description"] = f"{event['description']} {' '.join(line.split())}".strip()
            continue
        row = FIELD_ROW.match(line)
        if row:
            if row_name:
                end_row()
            row_name = row.group(1)
            line = row.group(2) or ""
        row_text.append(line.strip())
    end_row()
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="ReadMeFirst_TechnicalLogInformation.pdf")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON file to write")
    args = parser.parse_args()
    if not args.pdf.exists():
        sys.exit(f"ReadMe not found: {args.pdf}")

    events = parse(page_lines(PdfReader(args.pdf)))
    for event in events.values():
        event["fields"] = {
            name: text
            for name, text in event["fields"].items()
            if not name.startswith("context_") and name != "timestamp_utc"
        }
    doc = {"source": args.pdf.name, "events": dict(sorted(events.items()))}
    args.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = sum(len(e["fields"]) for e in events.values())
    print(f"{len(events)} event types and {fields:,} field descriptions written to {args.out}")


if __name__ == "__main__":
    main()
