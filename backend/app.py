import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import AzureOpenAI
from dotenv import load_dotenv
from retriever import retrieve_context

load_dotenv()

# -----------------------------------------------
# CONFIG
# -----------------------------------------------

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT") or "ktu-qp-finetuned"
SYLLABUS_DIR = os.path.join(os.path.dirname(__file__), "..", "syllabuses")

SUBJECT_KEYWORD_MAP: dict[str, str] = {
    "Python": "python",
    "Electrical and Electronics": "electrical",
    "Chemistry": "chemistry",
    "Physics": "physics",
    "Programming in C": "prog_c",
    "Foundations of Computing": "foundations",
    "Engineering Entrepreneurship and IPR": "entrepreneur",
}

LATEX_SUBJECTS = {
    "Physics",
    "Chemistry",
    "Electrical and Electronics",
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
# AZURE CLIENT
# -----------------------------------------------

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or "",
    api_key=os.getenv("AZURE_OPENAI_KEY") or "",
    api_version="2024-02-01",
)

# -----------------------------------------------
# FLASK APP
# -----------------------------------------------

app = Flask(__name__)
CORS(app)

SUPPORTED_SUBJECTS = [
    "Python",
    "Electrical and Electronics",
    "Chemistry",
    "Physics",
    "Programming in C",
    "Foundations of Computing",
    "Engineering Entrepreneurship and IPR",
]

SUPPORTED_SEMESTERS = ["1 & 2"]


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
        return jsonify({"error": "Semester must be '1 & 2'."}), 400

    try:
        context = retrieve_context(subject)
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    system_prompt = build_system_prompt(subject)

    user_prompt = f"""Subject: {subject} | Semester: {semester}
Slot-module mapping: Q1,2,9,10 → Module 1 | Q3,4,11,12 → Module 2 | Q5,6,13,14 → Module 3 | Q7,8,15,16 → Module 4

Context (use for question depth):
{context}"""

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2500,
            temperature=0.75,
        )
        qp_text = (response.choices[0].message.content or "").strip()
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
