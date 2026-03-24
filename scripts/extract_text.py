import fitz  # pymupdf
import pickle
import os
import re

# -----------------------------------------------
# CONFIG
# -----------------------------------------------

RAW_DIR = "data/raw"
OUTPUT_PATH = "data/processed/chunks.pkl"
SYLLABUS_DIR = "syllabuses"
CHUNK_SIZE = 300  # words per chunk
OVERLAP = 50  # word overlap between chunks

# Maps filename keywords → subject names.
# Keywords matched as whole word segments (split on _ and -) to avoid
# false matches e.g. "physics" inside "biophysics".
#
# REQUIRED NAMING CONVENTION for all PDFs in data/raw/:
#   Notes:        <keyword>_notes.pdf       e.g. python_notes.pdf
#   Module notes: <keyword>_mod1.pdf        e.g. python_mod1.pdf
#   Syllabus:     <keyword>_syllabus.pdf    e.g. python_syllabus.pdf
#   Past QPs:     <keyword>_qp_<year>.pdf   e.g. python_qp_2023.pdf
#
# Keywords per subject:
#   maths1, python, electrical, graphics, chemistry, physics,
#   maths2, prog_c, foundations, entrepreneur, discrete,
#   itworkshop, lspc, health
SUBJECT_MAP = {
    "python": "Python",
    "electrical": "Electrical and Electronics",
    "chemistry": "Chemistry",
    "physics": "Physics",
    "prog_c": "Programming in C",
    "foundations": "Foundations of Computing",
    "entrepreneur": "Engineering Entrepreneurship and IPR",
    "data_structures": "Data Structures and Algorithms",
    "oop_java": "Object Oriented Programming"
}

# Maps subject names → keyword used in their syllabus filename
SUBJECT_KEYWORD_MAP: dict[str, str] = {
    "Python": "python",
    "Electrical and Electronics": "electrical",
    "Chemistry": "chemistry",
    "Physics": "physics",
    "Programming in C": "prog_c",
    "Foundations of Computing": "foundations",
    "Engineering Entrepreneurship and IPR": "entrepreneur",
    "Data Structures and Algorithms": "data_structures",
    "Object Oriented Programming": "oop_java"
}

# -----------------------------------------------
# SYLLABUS KEYWORD INDEX
# -----------------------------------------------


def find_syllabus_file(subject: str) -> str | None:
    """Find the syllabus .txt file for a subject by keyword matching."""
    keyword = SUBJECT_KEYWORD_MAP.get(subject, "").lower()
    if not keyword or not os.path.isdir(SYLLABUS_DIR):
        return None
    for fname in os.listdir(SYLLABUS_DIR):
        if fname.endswith(".txt") and keyword in fname.lower():
            return os.path.join(SYLLABUS_DIR, fname)
    return None


def load_syllabus_keyword_index(subject: str) -> dict[int, list[str]]:
    """
    Load the syllabus txt for a subject and build a keyword index:
    { module_number: [keyword, keyword, ...] }

    Keywords are significant words (>=5 chars) extracted from each module's
    topic lines. Used to assign module numbers to chunks whose filenames
    don't have a mod tag.
    """
    path = find_syllabus_file(subject)
    if not path:
        return {}

    index: dict[int, list[str]] = {}
    current_mod: int | None = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"^MODULE\s+(\d):", line)
            if m:
                current_mod = int(m.group(1))
                index[current_mod] = []
                continue
            if current_mod is not None and line.startswith("-"):
                words = re.findall(r"\b[a-zA-Z]{5,}\b", line)
                index[current_mod].extend(w.lower() for w in words)

    return index


def detect_module_from_chunk(
    chunk: str,
    keyword_index: dict[int, list[str]],
) -> int | None:
    """
    Score a chunk against each module's keyword list.
    Return the module with the highest keyword hit count,
    or None if no module scores above zero.
    """
    if not keyword_index:
        return None

    chunk_lower = chunk.lower()
    scores: dict[int, int] = {}

    for mod_num, keywords in keyword_index.items():
        score = sum(1 for kw in keywords if kw in chunk_lower)
        scores[mod_num] = score

    best_mod = max(scores, key=lambda k: scores[k])
    return best_mod if scores[best_mod] > 0 else None


# -----------------------------------------------
# HELPERS
# -----------------------------------------------


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += str(page.get_text("text"))
    doc.close()
    return full_text


def clean_text(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP
) -> list[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def detect_subject(filename: str) -> str:
    """
    Detect subject from filename using whole-word keyword matching.
    Splits on _ and - to avoid partial matches.
    e.g. "prog_c_notes.pdf" → "Programming in C"
         "biophysics_notes.pdf" → "Unknown Subject" (not Physics)
    """
    stem = os.path.splitext(filename)[0].lower()
    parts = re.split(r"[_\-]", stem)
    parts_joined = "_" + "_".join(parts) + "_"
    for keyword, subject in SUBJECT_MAP.items():
        if f"_{keyword}_" in parts_joined:
            return subject
    return "Unknown Subject"


def detect_module_from_filename(filename: str) -> int | None:
    """Detect module number from filename e.g. python_mod1.pdf → 1."""
    match = re.search(r"mod(\d)", filename.lower())
    if match:
        return int(match.group(1))
    return None


def detect_doc_type(filepath: str) -> str:
    """Detect whether file is a syllabus, notes, or previous QP."""
    path_lower = filepath.lower()
    if "syllabus" in path_lower:
        return "syllabus"
    elif "previous_qp" in path_lower or "_qp_" in path_lower:
        return "previous_qp"
    return "notes"


# -----------------------------------------------
# MAIN
# -----------------------------------------------


def process_all_pdfs() -> None:
    all_chunks: list[str] = []
    all_metadata: list[dict] = []

    os.makedirs("data/processed", exist_ok=True)

    # Pre-load keyword indexes for ALL subjects that have syllabus files.
    # This is fully automatic — no hardcoding needed. Any subject with a
    # syllabus file in syllabuses/ will get keyword-based module tagging.
    keyword_indexes: dict[str, dict[int, list[str]]] = {}
    for subject in SUBJECT_MAP.values():
        idx = load_syllabus_keyword_index(subject)
        if idx:
            keyword_indexes[subject] = idx
            print(f"Loaded keyword index for '{subject}': modules {sorted(idx.keys())}")

    if not keyword_indexes:
        print(
            "WARNING: No syllabus files found in syllabuses/. Module tagging will be filename-only."
        )
        print("Run scripts/extract_syllabus.py to generate syllabus files first.")

    print()

    for root, dirs, files in os.walk(RAW_DIR):
        for filename in files:
            if not filename.lower().endswith(".pdf"):
                continue

            filepath = os.path.join(root, filename)
            print(f"Processing: {filepath}")

            raw_text = extract_text_from_pdf(filepath)
            text = clean_text(raw_text)

            if not text:
                print(f"  WARNING: No text extracted from {filename}, skipping.")
                continue

            subject = detect_subject(filename)
            filename_module = detect_module_from_filename(filename)
            doc_type = detect_doc_type(filepath)

            chunks = chunk_text(text)

            keyword_assigned = 0
            filename_assigned = 0
            unassigned = 0

            for chunk in chunks:
                if filename_module is not None:
                    module = filename_module
                    filename_assigned += 1
                elif subject in keyword_indexes:
                    module = detect_module_from_chunk(chunk, keyword_indexes[subject])
                    if module is not None:
                        keyword_assigned += 1
                    else:
                        unassigned += 1
                else:
                    module = None
                    unassigned += 1

                all_chunks.append(chunk)
                all_metadata.append(
                    {
                        "subject": subject,
                        "module": module,
                        "doc_type": doc_type,
                        "source": filename,
                    }
                )

            print(
                f"  Subject: {subject} | Type: {doc_type} | Chunks: {len(chunks)} "
                f"| filename-tagged: {filename_assigned} "
                f"| keyword-tagged: {keyword_assigned} "
                f"| untagged: {unassigned}"
            )

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump((all_chunks, all_metadata), f)

    print(f"\nDone. {len(all_chunks)} chunks saved to {OUTPUT_PATH}")
    print("Next step: run scripts/build_index.py to rebuild the FAISS index.")


if __name__ == "__main__":
    process_all_pdfs()
