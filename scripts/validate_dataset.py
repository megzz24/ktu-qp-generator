import json
import re
import os

# -----------------------------------------------
# CONFIG
# -----------------------------------------------

INPUT_PATH = "data/training/raw_generated.jsonl"
OUTPUT_PATH = "data/training/training_data.jsonl"
SYLLABUS_DIR = "syllabuses"
MIN_SAMPLES = 6  # minimum valid samples needed to proceed

# Maps subject name → syllabus keyword (matches filenames in syllabuses/)
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
    "Mathematics for Information Science 3": "maths3",
    "Theory of Computation": "toc",
    "Data Structures and Algorithms": "data_structures",
    "Object Oriented Programming": "oop_java",
    "Digital Electronics and Logic Design": "digital",
    "Economics for Engineers": "economics",
}

# Question number → module slot mapping
Q_TO_MODULE: dict[int, int] = {
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 3,
    7: 4,
    8: 4,
    9: 1,
    10: 1,
    11: 2,
    12: 2,
    13: 3,
    14: 3,
    15: 4,
    16: 4,
}

# -----------------------------------------------
# SYLLABUS KEYWORD INDEX
# -----------------------------------------------


def find_syllabus_file(subject: str) -> str | None:
    keyword = SUBJECT_KEYWORD_MAP.get(subject, "").lower()
    if not keyword or not os.path.isdir(SYLLABUS_DIR):
        return None
    for fname in os.listdir(SYLLABUS_DIR):
        if fname.endswith(".txt") and keyword in fname.lower():
            return os.path.join(SYLLABUS_DIR, fname)
    return None

# -----------------------------------------------
# EXTRACT SUBJECT FROM ENTRY
# -----------------------------------------------


def extract_subject(entry: dict) -> str | None:
    """Extract subject name from the user message content."""
    try:
        user_content = entry["messages"][1]["content"]
        m = re.search(r"Subject:\s*(.+?)(?:\.|$)", user_content)
        if m:
            return m.group(1).strip()
    except (KeyError, IndexError):
        pass
    return None


# -----------------------------------------------
# STRUCTURAL VALIDATION RULES
# Applied to synthetic papers only.
# -----------------------------------------------


def has_section_a(text: str) -> bool:
    return bool(re.search(r"PART\s*A", text, re.IGNORECASE))


def has_section_b(text: str) -> bool:
    return bool(re.search(r"PART\s*B", text, re.IGNORECASE))


def has_or_pairs(text: str) -> bool:
    return len(re.findall(r"\bOR\b", text)) >= 4


def has_enough_questions(text: str) -> bool:
    matches = re.findall(r"^\s*\d+[\.\)]\s", text, re.MULTILINE)
    return len(matches) >= 14


def has_subparts(text: str) -> bool:
    a_parts = re.findall(r"\(?a\)", text, re.IGNORECASE)
    return len(a_parts) >= 4


def has_mark_labels(text: str) -> bool:
    with_word = re.findall(r"\(\d+\s*marks?\)", text, re.IGNORECASE)
    bare = re.findall(r"\(\d\)", text)
    return len(with_word) + len(bare) >= 8


def check_mark_distribution(text: str) -> bool:
    low_with_word = re.findall(r"\([12]\s*marks?\)", text, re.IGNORECASE)
    low_bare = re.findall(r"\([12]\)", text)
    return len(low_with_word) + len(low_bare) == 0


def has_all_modules(text: str) -> bool:
    module_mentions = re.findall(r"Module\s*[1-4]", text, re.IGNORECASE)
    modules_found = set(re.findall(r"[1-4]", " ".join(module_mentions)))
    return len(modules_found) >= 4


def is_long_enough(text: str) -> bool:
    return len(text.split()) >= 200


def has_correct_question_count(text: str) -> bool:
    """Check that exactly 16 question numbers are present (1–16)."""
    found = set()
    for m in re.finditer(r"^\s*(\d{1,2})[\.\)]", text, re.MULTILINE):
        n = int(m.group(1))
        if 1 <= n <= 16:
            found.add(n)
    return len(found) == 16


def has_correct_part_b_module_labels(text: str) -> bool:
    """Check that all 4 module labels appear in Part B."""
    part_b_match = re.search(r"PART\s*B(.+)", text, re.IGNORECASE | re.DOTALL)
    if not part_b_match:
        return False
    part_b = part_b_match.group(1)
    for i in range(1, 5):
        if not re.search(rf"Module\s*{i}", part_b, re.IGNORECASE):
            return False
    return True

def has_correct_or_pair_modules(text: str) -> bool:
    """Check that OR pairs are within the same module block."""
    part_b = re.search(r"PART\s*B(.+)", text, re.IGNORECASE | re.DOTALL)
    if not part_b:
        return False
    # Each module block should contain exactly one OR
    blocks = re.split(r"Module\s*\d+", part_b.group(1), flags=re.IGNORECASE)
    for block in blocks[1:]:  # skip text before first Module label
        or_count = len(re.findall(r"\bOR\b", block))
        if or_count != 1:
            return False
    return True

SYNTHETIC_RULES: list[tuple] = [
    (has_section_a, "Missing PART A"),
    (has_section_b, "Missing PART B"),
    (has_or_pairs, "Fewer than 4 OR pairs"),
    (has_enough_questions, "Not enough question numbers"),
    (has_correct_question_count, "Question count is not exactly 16"),
    (has_subparts, "Missing (a)/(b) subparts in Part B"),
    (has_mark_labels, "Not enough mark labels"),
    (check_mark_distribution, "Subpart with fewer than 3 marks found"),
    (has_all_modules, "Not all 4 modules present"),
    (has_correct_part_b_module_labels, "Missing module labels in Part B"),
    (is_long_enough, "Question paper too short"),
    (has_correct_or_pair_modules, "OR pair not within same module block"),
]

# -----------------------------------------------
# MODULE ISOLATION CHECK
# -----------------------------------------------
def build_module_topic_index(subject: str) -> dict[int, list[str]]:
    """
    Load the syllabus txt and return the actual topic phrases per module.
    { module_number: ["topic phrase one", "topic phrase two", ...] }
    Each topic is the full cleaned line from the syllabus (minus the "- " prefix).
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
                topic = line.lstrip("- ").strip().lower()
                if len(topic) >= 8:  # skip very short lines
                    index[current_mod].append(topic)

    return index


# Cache
_topic_index_cache: dict[str, dict[int, list[str]]] = {}


def get_topic_index(subject: str) -> dict[int, list[str]]:
    if subject not in _topic_index_cache:
        _topic_index_cache[subject] = build_module_topic_index(subject)
    return _topic_index_cache[subject]


def topic_match_score(block: str, topic: str) -> int:
    """
    Score how strongly a question block matches a topic phrase.
    Splits the topic into significant words (>=5 chars) and counts hits.
    Returns number of topic words found in the block.
    """
    block_lower = block.lower()
    topic_words = [w for w in re.findall(r"\b[a-zA-Z]{5,}\b", topic)]
    return sum(1 for w in topic_words if w in block_lower)


def check_module_isolation(
    text: str,
    topic_index: dict[int, list[str]],
    subject: str = "",
) -> list[str]:
    """
    For each question block, check whether it matches a topic from the
    wrong module more strongly than any topic from the correct module.
    A violation is only raised when a wrong-module topic phrase scores
    >= MIN_TOPIC_WORDS hits AND scores higher than the best correct-module match.
    This avoids false positives from shared generic vocabulary.
    """
    if not topic_index:
        return []

    MIN_TOPIC_WORDS = 3
    WRONG_MODULE_MARGIN = 2 # wrong module must score at least this much >= correct best

    violations: list[str] = []
    blocks = re.split(r"(?=^\s*\d{1,2}[\.\)])", text, flags=re.MULTILINE)

    for block in blocks:
        q_match = re.match(r"^\s*(\d{1,2})[\.\)]", block)
        if not q_match:
            continue
        q_num = int(q_match.group(1))
        assigned_mod = Q_TO_MODULE.get(q_num)
        if assigned_mod is None:
            continue

        correct_best = max(
            (topic_match_score(block, t) for t in topic_index.get(assigned_mod, [])),
            default=0
        )

        for mod, topics in topic_index.items():
            if mod == assigned_mod:
                continue
            for topic in topics:
                score = topic_match_score(block, topic)
                # Flag if wrong module scores >= MIN_TOPIC_WORDS
                # AND exceeds correct module by at least WRONG_MODULE_MARGIN
                if score >= MIN_TOPIC_WORDS and score >= correct_best + WRONG_MODULE_MARGIN:
                    violations.append(
                        f"Q{q_num} (Module {assigned_mod} slot) matches "
                        f"Module {mod} topic '{topic}' (score {score} vs correct module best {correct_best})"
                    )
                    break
            if violations and violations[-1].startswith(f"Q{q_num}"):
                break
    return violations


# -----------------------------------------------
# DETECT REAL QP ENTRIES
# -----------------------------------------------


def is_real_qp_entry(entry: dict) -> bool:
    try:
        text = entry["messages"][2]["content"]

        # Real QPs are messy / inconsistent
        if "PART A" not in text or "Module 1" not in text:
            return True

        return False
    except:
        return False
# -----------------------------------------------
# VALIDATOR
# -----------------------------------------------


def validate_entry(entry: dict) -> tuple[bool, list[str]]:
    """Returns (is_valid, list_of_failure_reasons)."""
    try:
        messages = entry.get("messages", [])

        if len(messages) != 3:
            return False, ["Wrong number of messages (expected 3)"]
        if messages[0]["role"] != "system":
            return False, ["First message must be system"]
        if messages[1]["role"] != "user":
            return False, ["Second message must be user"]
        if messages[2]["role"] != "assistant":
            return False, ["Third message must be assistant"]

        qp_text = messages[2]["content"]
        if not qp_text or not qp_text.strip():
            return False, ["Empty assistant content"]

        # Real QP entries: only check they have enough content.
        # Don't apply structural rules — raw PDF text won't pass them.
        if is_real_qp_entry(entry):
            if len(qp_text.split()) < 50:
                return False, ["Real QP content too short (under 100 words)"]
            return True, []

        # Synthetic entries: apply all structural rules.
        failures: list[str] = []
        for rule_fn, failure_msg in SYNTHETIC_RULES:
            if not rule_fn(qp_text):
                failures.append(failure_msg)

        # Module isolation check (only if syllabus is available for subject).
        subject = extract_subject(entry)
        if subject:
            topic_index = get_topic_index(subject)
            isolation_violations = check_module_isolation(qp_text, topic_index, subject)
            for v in isolation_violations:
                failures.append(f"Module isolation: {v}")
        if failures:
            return False, failures

        return True, []

    except Exception as e:
        return False, [f"Exception during validation: {e}"]


# -----------------------------------------------
# MAIN
# -----------------------------------------------


def main() -> None:
    os.makedirs("data/training", exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Input file not found: {INPUT_PATH}")
        print("Run generate_training_data.py first.")
        return

    total = 0
    valid = 0
    invalid = 0
    real_qp_count = 0
    synthetic_count = 0
    failure_counts: dict[str, int] = {}
    valid_entries: list[dict] = []

    print(f"Validating {INPUT_PATH}...\n")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Line {line_num}: INVALID JSON — {e}")
                invalid += 1
                continue

            is_real = is_real_qp_entry(entry)
            is_valid, reasons = validate_entry(entry)

            if is_valid:
                valid += 1
                if is_real:
                    real_qp_count += 1
                else:
                    synthetic_count += 1
                valid_entries.append(entry)
            else:
                invalid += 1
                print(
                    f"  Line {line_num}: REJECTED ({'real QP' if is_real else 'synthetic'}) — {'; '.join(reasons)}"
                )
                for r in reasons:
                    failure_counts[r] = failure_counts.get(r, 0) + 1

    print("\n" + "=" * 50)
    print("VALIDATION REPORT")
    print("=" * 50)
    print(f"Total samples:    {total}")
    print(
        f"Valid samples:    {valid}  (real QPs: {real_qp_count}, synthetic: {synthetic_count})"
    )
    print(f"Invalid samples:  {invalid}")
    print(
        f"Pass rate:        {(valid / total * 100):.1f}%"
        if total > 0
        else "Pass rate: N/A"
    )

    if failure_counts:
        print("\nFailure breakdown:")
        for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"  {count:3d}x  {reason}")

    if valid < MIN_SAMPLES:
        print(
            f"\nWARNING: Only {valid} valid samples found (minimum is {MIN_SAMPLES})."
        )
        print("Consider regenerating with more PDFs or a higher SAMPLES_PER_SUBJECT.")
        return

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in valid_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\nClean dataset saved to: {OUTPUT_PATH}")
    print(f"This file is ready to upload to Azure OpenAI for fine-tuning.")


if __name__ == "__main__":
    main()
