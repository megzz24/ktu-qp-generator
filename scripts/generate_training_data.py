"""
generate_training_data.py
--------------------------
Generates synthetic KTU question papers for fine-tuning.

IMPORTANT: BASE_SYSTEM_PROMPT here must be IDENTICAL to backend/app.py.
The model is fine-tuned with this prompt — any difference degrades accuracy.

Usage:
    python scripts/generate_training_data.py
"""

import os
import re
import json
import pickle
import faiss
import fitz  # pymupdf
from google import genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from validate_dataset import validate_entry

load_dotenv()

# -----------------------------------------------
# CONFIG
# -----------------------------------------------

INDEX_PATH = "faiss/ktu_index.faiss"
META_PATH = "faiss/ktu_index_meta.pkl"
OUTPUT_PATH = "data/training/raw_generated.jsonl"
SYLLABUS_DIR = "syllabuses"
RAW_DIR = "data/raw"

SAMPLES_PER_SUBJECT = 10

SUBJECTS = [
    # S1 & S2
    ("Mathematics for Information Science 1", "1 & 2", "GAMAT101_maths1.txt"),
    ("Physics for Information Science", "1 & 2", "GAPHT121_physics.txt"),
    ("Chemistry for Information Science", "1 & 2", "GXCYT122_chemistry.txt"),
    (
        "Introduction to Electrical & Electronics Engineering",
        "1 & 2",
        "GXEST104_electrical.txt",
    ),
    ("Algorithmic Thinking with Python", "1 & 2", "UCEST105_python.txt"),
    ("Mathematics for Information Science 2", "1 & 2", "GAMAT201_maths2.txt"),
    ("Foundations of Computing", "1 & 2", "GXEST203_foundations.txt"),
    ("Programming in C", "1 & 2", "GXEST204_prog_c.txt"),
    ("Discrete Mathematics", "1 & 2", "PCCST205_discrete.txt"),
    ("Engineering Entrepreneurship and IPR", "1 & 2", "UCEST206_entrepreneur.txt"),
    # S3
    ("Mathematics for Information Science 3", "3", "GAMAT301_maths3.txt"),
    ("Theory of Computation", "3", "PCCST302_toc.txt"),
    ("Data Structures and Algorithms", "3", "PCCST303_data_structures.txt"),
    ("Object Oriented Programming", "3", "PBCST304_oop_java.txt"),
    ("Digital Electronics and Logic Design", "3", "GAEST305_digital.txt"),
    ("Economics for Engineers", "3", "UCHUT346_economics.txt"),
]

# -----------------------------------------------
# SYSTEM PROMPT
# MUST BE IDENTICAL TO backend/app.py BASE_SYSTEM_PROMPT
# -----------------------------------------------

BASE_SYSTEM_PROMPT = """
You are a KTU (APJ Abdul Kalam Technological University) question paper setter.

CRITICAL: YOUR OUTPUT MUST STRICTLY MATCH THE TEMPLATE BELOW.
If ANY rule is violated, the output is INVALID.

---

## MANDATORY STRUCTURE RULES (NON-NEGOTIABLE)

1. TOTAL QUESTIONS:

* EXACTLY 16 questions numbered from 1 to 16
* NO missing or extra numbers

2. PART A:

* EXACTLY 8 questions (Q1–Q8)
* Each carries (3) marks
* 2 questions per module
* NO module labels in Part A

3. PART B:

* EXACTLY 8 questions (Q9–Q16)
* MUST be grouped into 4 modules
* Each module MUST appear exactly once
* Each module MUST contain exactly 2 questions

4. OR PAIRS:

* EXACTLY 4 OR pairs (one per module)
* "OR" must appear on its own line
* Each OR pair must be within the SAME module

5. MARKS:

* Each Part B question totals 9 marks
* Preferred splits: (5)+(4) or (6)+(3)
* Single (9) allowed at most twice

6. MODULE ISOLATION (STRICT):

* Module 1 → Q1,2,9,10 ONLY
* Module 2 → Q3,4,11,12 ONLY
* Module 3 → Q5,6,13,14 ONLY
* Module 4 → Q7,8,15,16 ONLY
* NEVER mix topics between modules
* ONLY use topics explicitly present in that module’s syllabus

7. DO NOT:

* Add explanations, headings, or commentary
* Change formatting
* Skip numbering
* Combine modules
* Output anything outside the template

---

YOU MUST COPY THIS TEMPLATE EXACTLY
Replace ONLY the question text.
-------------------------------

PART A
(Answer all questions. Each question carries 3 marks)

1. [Module 1 question] (3)
2. [Module 1 question] (3)
3. [Module 2 question] (3)
4. [Module 2 question] (3)
5. [Module 3 question] (3)
6. [Module 3 question] (3)
7. [Module 4 question] (3)
8. [Module 4 question] (3)

PART B
(Answer any one full question from each module. Each question carries 9 marks)

Module 1

9. a) [Module 1 question] (5)
   b) [Module 1 question] (4)
   OR
10. a) [Module 1 question] (6)
    b) [Module 1 question] (3)

Module 2

11. a) [Module 2 question] (5)
    b) [Module 2 question] (4)
    OR
12. a) [Module 2 question] (6)
    b) [Module 2 question] (3)

Module 3

13. a) [Module 3 question] (5)
    b) [Module 3 question] (4)
    OR
14. a) [Module 3 question] (6)
    b) [Module 3 question] (3)

Module 4

15. a) [Module 4 question] (5)
    b) [Module 4 question] (4)
    OR
16. a) [Module 4 question] (6)
    b) [Module 4 question] (3)

---

## FINAL CHECK BEFORE OUTPUT (MANDATORY)

Before finishing, VERIFY:

* There are EXACTLY 16 questions
* Numbering is correct (1–16)
* All 4 modules are present
* Each module has exactly 2 questions
* There are exactly 4 OR pairs
* No module mixing has occurred

If ANY check fails → FIX before outputting."""

LATEX_ADDON = """
MATH FORMATTING: Use LaTeX for all expressions.
- Inline: $expr$ e.g. $x^2 + 3x = 0$
- Display: $$expr$$ e.g. $$\\int_0^\\infty e^{-x}dx = 1$$
- Use \\frac{a}{b}, x^n, x_n, \\int, \\sum, \\alpha, \\beta, \\theta, \\lambda for standard notation."""

LATEX_SUBJECTS = {
    "Physics for Information Science",
    "Chemistry for Information Science",
    "Introduction to Electrical & Electronics Engineering",
    "Mathematics for Information Science 1",
    "Mathematics for Information Science 2",
    "Mathematics for Information Science 3",
    "Discrete Mathematics",
    "Digital Electronics and Logic Design",
}


def build_system_prompt_with_syllabus(
    subject: str, syllabus_text: str, module_topics: dict[int, str]
) -> str:
    """Build full system prompt with syllabus appended. Mirrors app.py structure."""
    prompt = BASE_SYSTEM_PROMPT
    if subject in LATEX_SUBJECTS:
        prompt += LATEX_ADDON
    if module_topics:
        prompt += f"\n\nSYLLABUS — {subject.upper()}:\n" + "=" * 60 + "\n"
        for mod_num in sorted(module_topics.keys()):
            q_slots = {
                1: "Q1,2,9,10",
                2: "Q3,4,11,12",
                3: "Q5,6,13,14",
                4: "Q7,8,15,16",
            }
            prompt += f"\nModule {mod_num} topics ({q_slots.get(mod_num, '')}):\n"
            prompt += module_topics[mod_num] + "\n"
        prompt += "=" * 60
    elif syllabus_text:
        prompt += f"\n\nSYLLABUS — {subject.upper()}:\n{syllabus_text}"
    return prompt


# -----------------------------------------------
# VARIATION HINTS
# -----------------------------------------------

VARIATION_HINTS = [
    "Focus on applied/practical questions requiring step-by-step problem solving.",
    "Include definition and theory-based questions testing conceptual understanding.",
    "Include questions on real-world applications and case studies.",
    "Focus on comparison questions contrasting two related concepts from the same module.",
    "Include questions requiring students to trace through an example and predict outcomes.",
    "Include numerical examples or worked calculations where relevant to the subject.",
    "Part A questions must test the harder boundary concepts of each module.",
    "Part A questions must test the foundational entry-level concepts of each module.",
    "Part B scenarios set in industry or corporate environments (startups, banks, logistics).",
    "Mix scenario types: some Part B questions academic, some industry, some social.",
]

# -----------------------------------------------
# SUBJECT KEYWORD MAP
# -----------------------------------------------

SUBJECT_KEYWORD_MAP = {
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


def detect_subject_from_filename(filename: str) -> str | None:
    stem = os.path.splitext(filename)[0].lower()
    parts = re.split(r"[_\-]", stem)
    parts_joined = "_" + "_".join(parts) + "_"
    for keyword, subject in SUBJECT_KEYWORD_MAP.items():
        if f"_{keyword}_" in parts_joined:
            return subject
    return None


def is_qp_file(filename: str) -> bool:
    f = filename.lower()
    return "_qp_" in f or "previous_qp" in f or f.startswith("qp_")


# -----------------------------------------------
# SYLLABUS LOADING
# -----------------------------------------------


def load_syllabus(filename: str) -> str:
    path = os.path.join(SYLLABUS_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Syllabus file not found: {path}\nRun scripts/extract_syllabus.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def parse_module_topics(syllabus_text: str) -> dict[int, str]:
    modules: dict[int, list[str]] = {}
    current_mod: int | None = None
    for line in syllabus_text.splitlines():
        line = line.strip()
        m = re.match(r"^MODULE\s+(\d):", line)
        if m:
            current_mod = int(m.group(1))
            modules[current_mod] = []
            continue
        if current_mod is not None and line.startswith("-"):
            modules[current_mod].append(line)
    return {k: "\n".join(v) for k, v in modules.items()}


# -----------------------------------------------
# GEMINI CLIENT  (replaces Azure OpenAI)
# -----------------------------------------------

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------------------
# RETRIEVAL
# -----------------------------------------------


def load_index() -> tuple:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        chunks, metadata = pickle.load(f)
    return index, chunks, metadata


def retrieve_context(index, chunks, metadata, subject: str, top_k: int = 4):
    results_by_module = {1: [], 2: [], 3: [], 4: []}

    for mod in [1, 2, 3, 4]:
        query = f"{subject} module {mod} syllabus concepts exam questions"
        query_vec = embedder.encode([query]).astype("float32")

        distances, indices = index.search(query_vec, top_k)

        for i in indices[0]:
            if i < 0 or i >= len(chunks):
                continue

            meta = metadata[i]

            if subject.lower() not in meta["subject"].lower():
                continue

            if meta.get("module") == mod:
                results_by_module[mod].append(chunks[i])

            if len(results_by_module[mod]) >= 1:
                break

    return {mod: "\n\n".join(results_by_module[mod]) for mod in results_by_module}

# -----------------------------------------------
# GENERATION
# -----------------------------------------------


def generate_question_paper(
    subject: str,
    semester: str,
    context_by_module: dict[int, str],
    module_topics: dict[int, str],
    variation_index: int,
    system_prompt: str,
) -> str:
    hint = VARIATION_HINTS[variation_index % len(VARIATION_HINTS)]

    user_prompt = f"""Subject: {subject} | Semester: {semester}
Style focus: {hint}
Slot-module mapping: Q1,2,9,10 → Module 1 | Q3,4,11,12 → Module 2 | Q5,6,13,14 → Module 3 | Q7,8,15,16 → Module 4

Module 1 context (ONLY for Q1,2,9,10):
{context_by_module[1]}

Module 2 context (ONLY for Q3,4,11,12):
{context_by_module[2]}

Module 3 context (ONLY for Q5,6,13,14):
{context_by_module[3]}

Module 4 context (ONLY for Q7,8,15,16):
{context_by_module[4]}"""
    full_prompt = f"""
    {system_prompt}
USER INPUT:
{user_prompt}
"""
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=full_prompt,
        config={"temperature": 0.3, "top_p": 0.8, "max_output_tokens": 4000}
    )
    return (response.text or "").strip()


# -----------------------------------------------
# REAL QP EXTRACTION
# -----------------------------------------------


def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += str(page.get_text("text"))
    doc.close()
    return re.sub(r"\s+", " ", text).strip()


def extract_real_qps(
    subjects_in_run: set[str], subject_prompts: dict[str, str]
) -> list[dict]:
    """Scan data/raw/ for past QP PDFs and convert them into training entries."""
    entries = []
    if not os.path.isdir(RAW_DIR):
        return entries

    for root, _, files in os.walk(RAW_DIR):
        for filename in files:
            if not filename.lower().endswith(".pdf"):
                continue
            if not is_qp_file(filename):
                continue
            subject = detect_subject_from_filename(filename)
            if subject is None or subject not in subjects_in_run:
                continue

            qp_text = extract_pdf_text(os.path.join(root, filename))
            if not qp_text or len(qp_text.split()) < 200:
                print(f"  SKIPPED (too short): {filename}")
                continue

            entry = {
                "messages": [
                    {"role": "system", "content": subject_prompts[subject]},
                    {
                        "role": "user",
                        "content": f"Generate a KTU question paper for Semester 1 & 2, Subject: {subject}.",
                    },
                    {"role": "assistant", "content": qp_text},
                ]
            }
            
            valid, reasons = validate_entry(entry)

            if not valid:
                print(f"REJECTED — {reasons[0]}")
                continue
            entries.append(entry)
            print(f"  Added real QP: {filename} ({len(qp_text.split())} words)")

    return entries


# -----------------------------------------------
# MAIN
# -----------------------------------------------


def main() -> None:
    os.makedirs("data/training", exist_ok=True)

    print("Loading FAISS index...")
    index, chunks, metadata = load_index()

    all_entries: list[dict] = []
    subject_prompts: dict[str, str] = {}
    subject_module_topics: dict[str, dict[int, str]] = {}

    for subject, semester, syllabus_filename in SUBJECTS:
        syllabus_text = load_syllabus(syllabus_filename)
        module_topics = parse_module_topics(syllabus_text)
        subject_module_topics[subject] = module_topics
        subject_prompts[subject] = build_system_prompt_with_syllabus(
            subject, syllabus_text, module_topics
        )
        print(f"Loaded syllabus for {subject}: modules {sorted(module_topics.keys())}")

    print()

    subjects_in_run = {s for s, _, _ in SUBJECTS}
    print("Scanning data/raw/ for past question paper PDFs...")
    real_entries = extract_real_qps(subjects_in_run, subject_prompts)
    all_entries.extend(real_entries * 2)
    print(f"Added {len(real_entries)} real QP(s).\n")

    total_synthetic = len(SUBJECTS) * SAMPLES_PER_SUBJECT
    print(
        f"Generating {total_synthetic} synthetic papers ({SAMPLES_PER_SUBJECT} per subject)...\n"
    )

    for subject, semester, syllabus_filename in SUBJECTS:
        print(f"Subject: {subject}")
        module_topics = subject_module_topics[subject]
        system_prompt = subject_prompts[subject]
        context_by_module = retrieve_context(index, chunks, metadata, subject)

        for i in range(SAMPLES_PER_SUBJECT):
            print(f"  Sample {i + 1}/{SAMPLES_PER_SUBJECT}...", end=" ", flush=True)
            try:
                qp_text = generate_question_paper(
                    subject=subject,
                    semester=semester,
                    context_by_module=context_by_module,
                    module_topics=module_topics,
                    variation_index=i,
                    system_prompt=system_prompt,
                )

                entry = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Generate a KTU question paper for Semester {semester}, Subject: {subject}.",
                        },
                        {"role": "assistant", "content": qp_text},
                    ]
                }
                valid, reasons = validate_entry(entry)

                if not valid:
                    print(f"REJECTED — {reasons[0]}")
                    continue

                all_entries.append(entry)
                print("OK")
            except Exception as e:
                print(f"FAILED — {e}")
                continue
        print()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDone. {len(all_entries)} total samples saved to {OUTPUT_PATH}")
    print(f"  Real QPs:   {len(real_entries)}")
    print(f"  Synthetic:  {len(all_entries) - len(real_entries)}")
    print("Next step: run validate_dataset.py, then submit fine-tuning job on Google Vertex AI.")


if __name__ == "__main__":
    main()