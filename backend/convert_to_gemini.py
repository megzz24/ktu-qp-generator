"""
convert_to_gemini_format.py
----------------------------
Converts your existing training_data.jsonl (Azure/OpenAI 3-message format)
into the format required by Google AI Studio for Gemini fine-tuning.

INPUT format (what you have):
  {"messages": [
    {"role": "system",    "content": "You are a KTU question paper setter..."},
    {"role": "user",      "content": "Generate a KTU question paper for..."},
    {"role": "assistant", "content": "PART A\n1. ..."}
  ]}

OUTPUT format (what Google AI Studio needs):
  {"text_input": "You are a KTU question paper setter...\n\nGenerate a KTU question paper for...",
   "output":     "PART A\n1. ..."}

Usage (run from project root):
    python scripts/convert_to_gemini_format.py

Output file: data/training/gemini_training_data.jsonl
Upload that file to: https://aistudio.google.com → Create tuned model
"""

import json
import os

# -----------------------------------------------
# CONFIG — change these paths if needed
# -----------------------------------------------
INPUT_PATH  = "data/training/training_data.jsonl"
OUTPUT_PATH = "data/training/gemini_training_data.jsonl"

# Google AI Studio enforces these limits
MAX_INPUT_CHARS  = 32000   # text_input max length
MAX_OUTPUT_CHARS = 8000    # output max length
MIN_OUTPUT_WORDS = 100     # skip entries with suspiciously short outputs


def convert(input_path: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total      = 0
    converted  = 0
    skipped    = 0
    skip_reasons: dict[str, int] = {}

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Line {line_num}: invalid JSON — {e}")
                skipped += 1
                skip_reasons["Invalid JSON"] = skip_reasons.get("Invalid JSON", 0) + 1
                continue

            messages = entry.get("messages", [])
            if len(messages) != 3:
                skipped += 1
                skip_reasons["Wrong message count"] = skip_reasons.get("Wrong message count", 0) + 1
                continue

            system_content    = messages[0].get("content", "").strip()
            user_content      = messages[1].get("content", "").strip()
            assistant_content = messages[2].get("content", "").strip()

            if not assistant_content:
                skipped += 1
                skip_reasons["Empty output"] = skip_reasons.get("Empty output", 0) + 1
                continue

            # Build text_input: system prompt + blank line + user message
            text_input = system_content + "\n\n" + user_content

            # Length guard — truncate input if needed, never truncate output
            if len(text_input) > MAX_INPUT_CHARS:
                text_input = text_input[:MAX_INPUT_CHARS]

            if len(assistant_content) > MAX_OUTPUT_CHARS:
                skipped += 1
                skip_reasons["Output too long"] = skip_reasons.get("Output too long", 0) + 1
                continue

            if len(assistant_content.split()) < MIN_OUTPUT_WORDS:
                skipped += 1
                skip_reasons["Output too short"] = skip_reasons.get("Output too short", 0) + 1
                continue

            gemini_entry = {
                "text_input": text_input,
                "output":     assistant_content,
            }

            fout.write(json.dumps(gemini_entry, ensure_ascii=False) + "\n")
            converted += 1

    # -----------------------------------------------
    # REPORT
    # -----------------------------------------------
    print()
    print("=" * 50)
    print("CONVERSION REPORT")
    print("=" * 50)
    print(f"Total input entries : {total}")
    print(f"Successfully converted: {converted}")
    print(f"Skipped             : {skipped}")
    if skip_reasons:
        print("\nSkip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"  {count:3d}x  {reason}")
    print()
    print(f"Output saved to: {output_path}")
    print()

    if converted < 10:
        print("⚠️  WARNING: Very few samples converted.")
        print("   Google AI Studio recommends at least 20 examples,")
        print("   ideally 100+. Run generate_training_data.py to create more.")
    elif converted < 20:
        print("⚠️  WARNING: Google AI Studio recommends at least 20 examples.")
        print("   Run generate_training_data.py with higher SAMPLES_PER_SUBJECT.")
    else:
        print(f"✅  {converted} samples ready.")
        print("Next step: upload gemini_training_data.jsonl to Google AI Studio.")
        print("  → https://aistudio.google.com")
        print("  → Click 'Create tuned model' → Upload this file")
        print("  → Base model: Gemini 1.5 Flash")
        print("  → After tuning, copy your model ID into backend/app.py")


if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found.")
        print("Run scripts/generate_training_data.py first.")
    else:
        convert(INPUT_PATH, OUTPUT_PATH)