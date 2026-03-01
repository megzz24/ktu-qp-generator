"""
extract_syllabus.py
-------------------
Extracts module-topic mapping from KTU syllabus PDFs and saves
a structured plain-text file to syllabuses/<course_code>.txt

Usage:
    python scripts/extract_syllabus.py \
        --pdf data/raw/syllabuses/python_syllabus.pdf \
        --course_code UCEST105 \
        --subject Python \
        --output syllabuses/UCEST105_python.txt

To add a new subject, just run with the new PDF and course code.
"""

import argparse
import os
import re
import fitz  # pymupdf


# -----------------------------------------------
# PDF EXTRACTION
# -----------------------------------------------


def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -----------------------------------------------
# MODULE EXTRACTION
# -----------------------------------------------


def extract_modules(raw_text: str) -> dict[int, dict]:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    
    modules: dict[int, dict] = {}
    current_mod: int | None = None

    for line in lines:
        # Match standalone module numbers (table cell: "1", "2", "3", "4")
        if re.match(r"^[1-4]$", line):
            current_mod = int(line)
            if current_mod not in modules:
                modules[current_mod] = {"title": f"Module {current_mod}", "topics": []}
            continue

        # Also match "Module 1", "Module 2" style headers (fallback)
        m = re.match(r"^module\s*(\d)\b", line, re.IGNORECASE)
        if m:
            current_mod = int(m.group(1))
            if current_mod not in modules:
                modules[current_mod] = {"title": f"Module {current_mod}", "topics": []}
            continue

        if current_mod is None:
            continue

        # Skip administrative lines
        if len(line) < 10:
            continue
        if re.search(
            r"(marks|examination|semester|credits|CIE|ESE|bloom|outcome|CO\d|PO\d|contact|hours|syllabus|course|references|text book|reference book|prerequisite)",
            line, re.IGNORECASE,
        ):
            continue
        if re.match(r"^\d+$", line):  # standalone numbers (contact hours)
            continue

        # First substantial line after module number = title
        if not modules[current_mod]["topics"] and modules[current_mod]["title"] == f"Module {current_mod}":
            modules[current_mod]["title"] = line[:60]

        modules[current_mod]["topics"].append(line)

    if not modules:
        raise ValueError("No module headers found in PDF. Check the PDF formatting.")

    # Deduplicate topics
    for mod in modules.values():
        seen: set[str] = set()
        deduped: list[str] = []
        for t in mod["topics"]:
            key = t[:60].lower()
            if key not in seen:
                seen.add(key)
                deduped.append(t)
        mod["topics"] = deduped

    return modules

# -----------------------------------------------
# OUTPUT FORMATTING
# -----------------------------------------------


def format_syllabus_file(
    course_code: str,
    subject: str,
    modules: dict[int, dict],
) -> str:
    lines: list[str] = [
        f"COURSE CODE: {course_code}",
        f"SUBJECT: {subject}",
        "=" * 60,
        "",
    ]

    for mod_num in sorted(modules):
        mod = modules[mod_num]
        lines.append(f"MODULE {mod_num}: {mod['title']}")
        lines.append("-" * 50)
        for topic in mod["topics"]:
            lines.append(f"  - {topic}")
        lines.append("")

    return "\n".join(lines)


# -----------------------------------------------
# MAIN
# -----------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract KTU syllabus module map from PDF."
    )
    parser.add_argument("--pdf", required=True, help="Path to syllabus PDF")
    parser.add_argument("--course_code", required=True, help="e.g. UCEST105")
    parser.add_argument("--subject", required=True, help="e.g. Python")
    parser.add_argument("--output", required=True, help="Output .txt path")
    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Extracting text from: {args.pdf}")
    raw_text = extract_pdf_text(args.pdf)

    print("Parsing module structure...")
    modules = extract_modules(raw_text)

    print(f"Found {len(modules)} modules:")
    for mod_num, mod in modules.items():
        print(f"  Module {mod_num}: {mod['title']} ({len(mod['topics'])} topic lines)")

    output = format_syllabus_file(args.course_code, args.subject, modules)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
