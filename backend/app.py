import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.generativeai as genai
from dotenv import load_dotenv
from retriever import retrieve_context

load_dotenv()

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")


SYLLABUS_DIR = os.path.join(os.path.dirname(__file__), "..", "syllabuses")

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

def find_syllabus_file(subject: str) -> str | None:
    keyword = SUBJECT_KEYWORD_MAP.get(subject, "").lower()
    if not keyword or not os.path.isdir(SYLLABUS_DIR):
        return None
    for fname in os.listdir(SYLLABUS_DIR):
        if fname.endswith(".txt") and keyword in fname.lower():
            return os.path.join(SYLLABUS_DIR, fname)
    return None


def load_syllabus(subject: str) -> str:
    path = find_syllabus_file(subject)
    if not path:
        return ""
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
# SYSTEM PROMPT
# NOTE: Must be IDENTICAL to generate_training_data.py — the fine-tuned
# model was trained with this exact structure.
# -----------------------------------------------
BASE_SYSTEM_PROMPT = """
You are a KTU (APJ Abdul Kalam Technological University) question paper setter.

CRITICAL: YOUR OUTPUT MUST STRICTLY MATCH THE TEMPLATE BELOW.
If ANY rule is violated, the output is INVALID.

DO NOT ADD ANY TEXT BEFORE OR AFTER THE TEMPLATE.
DO NOT INCLUDE EXPLANATIONS, HEADINGS, OR EXTRA NOTES.
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


def build_system_prompt(subject: str) -> str:
    syllabus = load_syllabus(subject)
    module_topics = parse_module_topics(syllabus) if syllabus else {}

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
    elif syllabus:
        prompt += f"\n\nSYLLABUS — {subject.upper()}:\n{syllabus}"

    return prompt





# -----------------------------------------------
# FLASK APP
# -----------------------------------------------

app = Flask(__name__)
CORS(app)

SUPPORTED_SUBJECTS = [
    # S1 & S2
    "Mathematics for Information Science 1",
    "Physics for Information Science",
    "Chemistry for Information Science",
    "Introduction to Electrical & Electronics Engineering",
    "Algorithmic Thinking with Python",
    "Mathematics for Information Science 2",
    "Foundations of Computing",
    "Programming in C",
    "Discrete Mathematics",
    "Engineering Entrepreneurship and IPR",
    # S3
    "Mathematics for Information Science 3",
    "Theory of Computation",
    "Data Structures and Algorithms",
    "Object Oriented Programming",
    "Digital Electronics and Logic Design",
    "Economics for Engineers",
]

SUPPORTED_SEMESTERS = ["1 & 2", "3"]

@app.route("/generate", methods=["POST"])
def generate() -> tuple:
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body received."}), 400

    subject: str = (data.get("subject") or "").strip()
    semester: str = (data.get("semester") or "1 & 2").strip()

    if not subject:
        return jsonify({"error": "Subject is required."}), 400
    if subject not in SUPPORTED_SUBJECTS:
        return jsonify({"error": f"Subject '{subject}' is not supported."}), 400
    if semester not in SUPPORTED_SEMESTERS:
        return jsonify({"error": f"Semester must be one of {SUPPORTED_SEMESTERS}."}), 400

    try:
        context_by_module = retrieve_context(subject)
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    system_prompt = build_system_prompt(subject)
    
    user_prompt = f"""Subject: {subject} | Semester: {semester}
Slot-module mapping: Q1,2,9,10 → Module 1 | Q3,4,11,12 → Module 2 | Q5,6,13,14 → Module 3 | Q7,8,15,16 → Module 4

Module 1 context (ONLY for Q1,2,9,10):
{context_by_module[1]}

Module 2 context (ONLY for Q3,4,11,12):
{context_by_module[2]}

Module 3 context (ONLY for Q5,6,13,14):
{context_by_module[3]}

Module 4 context (ONLY for Q7,8,15,16):
{context_by_module[4]}"""

    try:
        full_prompt = system_prompt + "\n\nUSER INPUT:\n" + user_prompt  # FIX: was undefined

        response = model.generate_content(
            full_prompt,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.8,
                "max_output_tokens": 2500
            }
        )

        qp_text = (response.text or "").strip()

    except Exception as e:
        return jsonify({"error": f"Model inference failed: {str(e)}"}), 500

    if not qp_text:
        return jsonify({"error": "Model returned an empty response."}), 500

    return (
        jsonify(
            {
                "question_paper": qp_text,
                "latex": subject in LATEX_SUBJECTS,
            }
        ),
        200,
    )


@app.route("/subjects", methods=["GET"])
def subjects() -> tuple:
    syllabus_status = {
        s: (find_syllabus_file(s) is not None) for s in SUPPORTED_SUBJECTS
    }
    return (
        jsonify(
            {
                "subjects": SUPPORTED_SUBJECTS,
                "semesters": SUPPORTED_SEMESTERS,
                "syllabus_loaded": syllabus_status,
            }
        ),
        200,
    )


@app.route("/health", methods=["GET"])
def health() -> tuple:
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)