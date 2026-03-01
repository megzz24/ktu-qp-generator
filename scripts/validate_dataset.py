import json
import re
import os

# -----------------------------------------------
# CONFIG
# -----------------------------------------------

INPUT_PATH = "data/training/raw_generated.jsonl"
OUTPUT_PATH = "data/training/training_data.jsonl"
MIN_SAMPLES = 10  # minimum valid samples needed to proceed

# -----------------------------------------------
# VALIDATION RULES
# Applied to synthetic papers only (not real past QPs).
# Real past QPs are validated leniently — just checked for
# non-empty content, since their raw PDF text won't pass
# structural checks (page numbers, CO labels, headers etc.)
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
    # Lowered from 500 to 300 — GPT-3.5 synthetic papers are valid but concise.
    # Real past QPs bypass this check entirely (see validate_entry).
    return len(text.split()) >= 300


SYNTHETIC_RULES: list[tuple] = [
    (has_section_a, "Missing PART A"),
    (has_section_b, "Missing PART B"),
    (has_or_pairs, "Fewer than 4 OR pairs"),
    (has_enough_questions, "Not enough question numbers"),
    (has_subparts, "Missing (a)/(b) subparts in Part B"),
    (has_mark_labels, "Not enough mark labels"),
    (check_mark_distribution, "Subpart with fewer than 3 marks found"),
    (has_all_modules, "Not all 4 modules present"),
    (is_long_enough, "Question paper too short"),
]

# -----------------------------------------------
# DETECT REAL QP ENTRIES
# Real QP entries are identified by the user message —
# they say exactly "Generate a KTU question paper for Semester 1 & 2, Subject: X."
# with no extra content (no style hint, no notes context).
# Synthetic entries have a longer user message with style hints and context.
# -----------------------------------------------


def is_real_qp_entry(entry: dict) -> bool:
    """
    Returns True if this training entry came from a real past QP PDF
    rather than being synthetically generated.
    Real QP user messages are short — just subject and semester.
    Synthetic ones contain style hints and notes context.
    """
    try:
        user_content = entry["messages"][1]["content"]
        # Real QP user prompts are short (under 30 words)
        return len(user_content.split()) < 30
    except (KeyError, IndexError):
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
            if len(qp_text.split()) < 100:
                return False, ["Real QP content too short (under 100 words)"]
            return True, []

        # Synthetic entries: apply all structural rules.
        failures = []
        for rule_fn, failure_msg in SYNTHETIC_RULES:
            if not rule_fn(qp_text):
                failures.append(failure_msg)

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
