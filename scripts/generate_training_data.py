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
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

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
    ("Python", "1 & 2", "UCEST105_python.txt"),
    ("Electrical and Electronics", "1 & 2", "UCEST102_electrical.txt"),
    ("Chemistry", "1 & 2", "UCEST104_chemistry.txt"),
    ("Physics", "1 & 2", "UCEST106_physics.txt"),
    ("Programming in C", "1 & 2", "UCEST108_prog_c.txt"),
    ("Foundations of Computing", "1 & 2", "UCEST109_foundations.txt"),
    ("Engineering Entrepreneurship and IPR", "1 & 2", "UCEST110_entrepreneur.txt"),
    ("Data Structures and Algorithms", "1 & 2","PCCST303_data_structures.txt"),
    ("Object Oriented Programming", "1 & 2","PBCST304_oop_java.txt")
]

# -----------------------------------------------
# SYSTEM PROMPT
# MUST BE IDENTICAL TO backend/app.py BASE_SYSTEM_PROMPT
# -----------------------------------------------

BASE_SYSTEM_PROMPT = """You are a KTU (APJ Abdul Kalam Technological University) question paper setter.

KTU FORMAT:
- 4 modules, each with fixed question slots — never mix topics across modules.

PART A (24 marks total):
- 8 questions, 3 marks each. 2 questions per module.
- Q1-2 → Module 1 | Q3-4 → Module 2 | Q5-6 → Module 3 | Q7-8 → Module 4
- No module labels in Part A. Number questions 1–8 only.
- Format: "1. Question text. (3)"

PART B (36 marks total):
- 8 questions in OR pairs, 9 marks each. 2 questions per module.
- Q9-10 → Module 1 | Q11-12 → Module 2 | Q13-14 → Module 3 | Q15-16 → Module 4
- Label each module before its OR pair: "Module 1", etc.
- "OR" appears on its own line between the two questions.
- Subpart splits (preferred): (5)+(4) or (6)+(3). Single-question format (9) allowed at most twice.
- Subpart format: "    a) Question text. (5)"

OUTPUT (no title, no preamble, start directly with PART A):

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

9.  a) [question] (5)
    b) [question] (4)
OR
10. a) [question] (6)
    b) [question] (3)

[Repeat pattern for Module 2 (Q11-12), Module 3 (Q13-14), Module 4 (Q15-16)]"""

LATEX_ADDON = """

MATH FORMATTING: Use LaTeX for all expressions.
- Inline: $expr$ e.g. $x^2 + 3x = 0$
- Display: $$expr$$ e.g. $$\\int_0^\\infty e^{-x}dx = 1$$
- Use \\frac{a}{b}, x^n, x_n, \\int, \\sum, \\alpha, \\beta, \\theta, \\lambda for standard notation."""

LATEX_SUBJECTS = {
    "Physics",
    "Chemistry",
    "Electrical and Electronics",
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
    "python": "Python",
    "electrical": "Electrical and Electronics",
    "chemistry": "Chemistry",
    "physics": "Physics",
    "prog_c": "Programming in C",
    "foundations": "Foundations of Computing",
    "entrepreneur": "Engineering Entrepreneurship and IPR",
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
# AZURE CLIENT
# -----------------------------------------------

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or "",
    api_key=os.getenv("AZURE_OPENAI_KEY") or "",
    api_version="2024-02-01",
)

GENERATION_MODEL = os.getenv("AZURE_OPENAI_BASE_DEPLOYMENT") or "gpt-4o-mini"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------------------
# RETRIEVAL
# -----------------------------------------------


def load_index() -> tuple:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        chunks, metadata = pickle.load(f)
    return index, chunks, metadata


def retrieve_context(
    index, chunks: list, metadata: list, subject: str, top_k: int = 8
) -> str:
    query = f"KTU {subject} important topics examination questions"
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k * 5)

    results_by_module: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}
    general_results: list[str] = []

    for i in indices[0]:
        if i < 0 or i >= len(chunks):
            continue
        meta = metadata[i]
        if subject.lower() not in meta["subject"].lower():
            continue
        mod = meta.get("module")
        if isinstance(mod, int) and mod in results_by_module:
            if len(results_by_module[mod]) < 2:
                results_by_module[mod].append(chunks[i])
        else:
            if len(general_results) < 2:
                general_results.append(chunks[i])

    context_parts: list[str] = []
    for mod in [1, 2, 3, 4]:
        if results_by_module[mod]:
            context_parts.append(f"--- Module {mod} content ---")
            context_parts.extend(results_by_module[mod])
    if general_results:
        context_parts.append("--- General content ---")
        context_parts.extend(general_results)

    return "\n\n".join(context_parts)


# -----------------------------------------------
# GENERATION
# -----------------------------------------------


def generate_question_paper(
    subject: str,
    semester: str,
    context: str,
    module_topics: dict[int, str],
    variation_index: int,
    system_prompt: str,
) -> str:
    hint = VARIATION_HINTS[variation_index % len(VARIATION_HINTS)]

    user_prompt = f"""Subject: {subject} | Semester: {semester}
Style focus: {hint}
Slot-module mapping: Q1,2,9,10 → Module 1 | Q3,4,11,12 → Module 2 | Q5,6,13,14 → Module 3 | Q7,8,15,16 → Module 4

Context (use for question depth):
{context}"""

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4000,
        temperature=0.75,
    )
    return (response.choices[0].message.content or "").strip()


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
    all_entries.extend(real_entries)
    print(f"Added {len(real_entries)} real QP(s).\n")

    total_synthetic = len(SUBJECTS) * SAMPLES_PER_SUBJECT
    print(
        f"Generating {total_synthetic} synthetic papers ({SAMPLES_PER_SUBJECT} per subject)...\n"
    )

    for subject, semester, syllabus_filename in SUBJECTS:
        print(f"Subject: {subject}")
        module_topics = subject_module_topics[subject]
        system_prompt = subject_prompts[subject]
        context = retrieve_context(index, chunks, metadata, subject)

        for i in range(SAMPLES_PER_SUBJECT):
            print(f"  Sample {i + 1}/{SAMPLES_PER_SUBJECT}...", end=" ", flush=True)
            try:
                qp_text = generate_question_paper(
                    subject=subject,
                    semester=semester,
                    context=context,
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
    print("Next step: run validate_dataset.py, then submit fine-tuning job on Azure.")


if __name__ == "__main__":
    main()
