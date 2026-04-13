import fitz  # pymupdf
import pickle
import os
import re
import pytesseract
from PIL import Image
import io

import platform

# Auto-detect Tesseract path (set manually if auto-detection fails)
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# On Linux/Mac, tesseract is usually on PATH — no need to set this

# -----------------------------------------------
# CONFIG
# -----------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data/raw")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/processed/chunks.pkl")
SYLLABUS_DIR = os.path.join(BASE_DIR, "syllabuses")

CHUNK_SIZE = 150  # words per chunk
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
    "maths1": "Mathematics for Information Science 1",
    "physics": "Physics for Information Science",
    "chemistry": "Chemistry for Information Science",
    "electrical": "Introduction to Electrical & Electronics Engineering",
    "python": "Algorithmic Thinking with Python",
    "maths2": "Mathematics for Information Science 2",
    "foundations": "Foundations of Computing",
    "prog_c": "Programming in C",
    "discrete": "Discrete Mathematics",
    "entrepreneur": "Engineering Entrepreneurship and IPR",
    # S3 additions
    "maths3": "Mathematics for Information Science 3",
    "data_structures": "Data Structures and Algorithms",
    "oop_java": "Object Oriented Programming",
    "toc": "Theory of Computation",
    "digital": "Digital Electronics and Logic Design",
    "economics": "Economics for Engineers",
}

# Maps subject names → keyword used in their syllabus filename
SUBJECT_KEYWORD_MAP: dict[str, str] = {
    "Algorithmic Thinking with Python": "python",
    "Introduction to Electrical & Electronics Engineering": "electrical",
    "Chemistry for Information Science": "chemistry",
    "Physics for Information Science": "physics",
    "Programming in C": "prog_c",
    "Foundations of Computing": "foundations",
    "Engineering Entrepreneurship and IPR": "entrepreneur",
    "Mathematics for Information Science 1": "maths1",
    "Mathematics for Information Science 2": "maths2",
    "Discrete Mathematics": "discrete",
    # S3 additions
    "Mathematics for Information Science 3": "maths3",
    "Data Structures and Algorithms": "data_structures",
    "Object Oriented Programming": "oop_java",
    "Theory of Computation": "toc",
    "Digital Electronics and Logic Design": "digital",
    "Economics for Engineers": "economics",
}

# -----------------------------------------------
# SYLLABUS KEYWORD INDEX
# -----------------------------------------------
def extract_text_with_ocr(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        text += pytesseract.image_to_string(img)

    doc.close()
    return text

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
                words = re.findall(r"\b[a-zA-Z]{3,}\b", line)
                index[current_mod].extend(w.lower() for w in words)

    return index

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
    text = re.sub(r"\s+", " ", text)
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

def detect_module_with_confidence(chunk, keyword_indexes):
    chunk_lower = chunk.lower()
    scores = {}

    for mod, keywords in keyword_indexes.items():
        score = sum(1 for kw in keywords if kw in chunk_lower)
        scores[mod] = score

    if not scores:
        return None, 0

    best_mod, best_score = max(scores.items(), key=lambda x: x[1])
    return best_mod, best_score

def extract_qp_by_module(text: str) -> dict[int, str]:
    """
    Extract text from a KTU QP grouped by module.
    Part A: Q1-2 → Mod1, Q3-4 → Mod2, Q5-6 → Mod3, Q7-8 → Mod4
    Part B: Split by "Module-N" / "Module N" labels
    Returns { 1: "combined text", 2: "...", 3: "...", 4: "..." }
    """
    modules: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}

    # -----------------------------------------------
    # PART A — fixed slot mapping
    # -----------------------------------------------
    part_a_match = re.search(
        r"PART\s*A(.+?)(?:PART\s*B|$)", text, re.IGNORECASE | re.DOTALL
    )
    if part_a_match:
        part_a = part_a_match.group(1)
        # Split into individual questions on number boundaries
        q_blocks = re.split(r"(?=^\s*\d{1,2}[\.\)]\s)", part_a, flags=re.MULTILINE)
        for block in q_blocks:
            q_match = re.match(r"^\s*(\d{1,2})[\.\)]", block)
            if not q_match:
                continue
            q_num = int(q_match.group(1))
            if 1 <= q_num <= 2:
                modules[1].append(block.strip())
            elif 3 <= q_num <= 4:
                modules[2].append(block.strip())
            elif 5 <= q_num <= 6:
                modules[3].append(block.strip())
            elif 7 <= q_num <= 8:
                modules[4].append(block.strip())

    # -----------------------------------------------
    # PART B — split by Module labels
    # -----------------------------------------------
    part_b_match = re.search(
        r"PART\s*B(.+)", text, re.IGNORECASE | re.DOTALL
    )
    if part_b_match:
        part_b = part_b_match.group(1)

        # Split on "Module-N", "Module N", "Module–N"
        module_splits = re.split(
            r"(?=module\s*[-–—:]?\s*[1-4]\s*:?)", part_b, flags=re.IGNORECASE
        )

        for section in module_splits:
            mod_match = re.match(
                r"module\s*[-–—:]?\s*([1-4])\s*:?", section, re.IGNORECASE
            )
            if not mod_match:
                continue
            mod_num = int(mod_match.group(1))
            # Strip the module label line itself
            content = re.sub(
                r"module\s*[-–—:]?\s*([1-4])\s*:?", "", section,
                count=1, flags=re.IGNORECASE
            ).strip()
            if content:
                modules[mod_num].append(content)

    return {
        mod: "\n\n".join(parts)
        for mod, parts in modules.items()
        if parts
    }

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

            # OCR fallback
            if len(text.split()) < 50:
                print(f"  OCR fallback triggered for {filename}")
                raw_text = extract_text_with_ocr(filepath)
                text = clean_text(raw_text)
            
            if not text.strip():
                print(f"  OCR failed for {filename}, skipping.")
                continue

            if not text:
                print(f"  WARNING: No text extracted from {filename}, skipping.")
                continue

            subject = detect_subject(filename)
            filename_module = detect_module_from_filename(filename)
            doc_type = detect_doc_type(filepath)

            keyword_assigned = 0
            filename_assigned = 0
            unassigned = 0
            processed_chunks = 0
            chunks: list[str] = []
            
            module = None
            
            if doc_type == "previous_qp":
                # Extract module-wise directly from QP structure
                module_text = extract_qp_by_module(text)
                
                if any(module_text.values()):
                    for mod_num, mod_text in module_text.items():
                        mod_chunks = chunk_text(mod_text)
                        for chunk in mod_chunks:
                            chunk = chunk.strip()
                            if len(chunk.split()) < 5:
                                continue
                            all_chunks.append(chunk)
                            all_metadata.append({
                                "subject": subject,
                                "module": mod_num,
                                "doc_type": doc_type,
                                "source": filename,
                            })
                            keyword_assigned += 1
                            processed_chunks += 1
                    print(
                        f"  Subject: {subject} | Type: {doc_type} | Chunks: {processed_chunks} "
                        f"| module-tagged (QP structure): {keyword_assigned} "
                        f"| untagged: {unassigned}"
                    )
                    continue  # skip the normal chunking loop below for this file

                # Fallback if module labels not found in QP
                print(f"  WARNING: No module labels found in {filename}, falling back to keyword tagging.")
                chunks = chunk_text(text)
            else:
                chunks = chunk_text(text)
                if len(chunks) <= 1:
                    chunks = re.split(r"\n+", raw_text)
                    chunks = [c.strip() for c in chunks if len(c.split()) > 20]

            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk.split()) < 5:
                    continue

                processed_chunks += 1
                module = None

                if doc_type == "previous_qp":
                    # Module is already determined by extract_qp_by_module
                    # handled separately below — skip normal chunking
                    pass
                elif filename_module is not None:
                    module = filename_module
                    filename_assigned += 1
                elif subject in keyword_indexes:
                    module_candidate, confidence = detect_module_with_confidence(
                        chunk, keyword_indexes[subject]
                    )
                    if confidence >= 3:
                        module = module_candidate
                        keyword_assigned += 1
                    else:
                        module = None
                        unassigned += 1
                else:
                    module = None
                    unassigned += 1

                all_chunks.append(chunk)
                all_metadata.append({
                    "subject": subject,
                    "module": module,
                    "doc_type": doc_type,
                    "source": filename,
                })
            print(
                f"  Subject: {subject} | Type: {doc_type} | Chunks: {processed_chunks} "
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