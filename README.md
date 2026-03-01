# KTU Question Paper Generator

An AI-powered system that generates KTU (APJ Abdul Kalam Technological University) B.Tech Semester 1 & 2 examination question papers. A fine-tuned GPT model handles question generation while a FAISS vector index retrieves subject-specific content from your own notes and past papers to produce deep, syllabus-accurate questions.

---

## How It Works

```
Your PDFs (notes + past QPs + syllabus)
        ↓
  extract_text.py           ← chunks PDFs, tags each chunk with subject + module
        ↓
  build_index.py            ← embeds chunks, builds FAISS vector index
        ↓
  generate_training_data.py ← generates synthetic question papers via GPT API
        ↓
  validate_dataset.py       ← filters out malformed papers
        ↓
  Azure Fine-Tuning          ← trains the model on your dataset
        ↓
  app.py (Flask API)         ← serves /generate requests at runtime
                               retriever.py fetches relevant chunks per request
```

At inference time: for every `/generate` request, `retriever.py` searches the FAISS index for relevant notes chunks (balanced across all 4 modules), injects them into the prompt alongside the full syllabus module map, and sends both to the fine-tuned model.

---

## Supported Subjects

| Subject | Filename Keyword | LaTeX Rendering |
|---|---|---|
| Python | `python` | No |
| Electrical and Electronics | `electrical` | Yes |
| Chemistry | `chemistry` | Yes |
| Physics | `physics` | Yes |
| Programming in C | `prog_c` | No |
| Foundations of Computing | `foundations` | No |
| Engineering Entrepreneurship and IPR | `entrepreneur` | No |

Subjects with LaTeX rendering return `"latex": true` in the API response. The frontend should pass the question paper text through MathJax or KaTeX when this flag is set.

---

## Project Structure

```
ktu-qp-generator/
│
├── backend/
│   ├── app.py                      ← Flask API — run this to serve requests
│   └── retriever.py                ← FAISS search, called on every /generate request
│
├── scripts/
│   ├── extract_syllabus.py         ← Step 1: parse syllabus PDFs into module map txt files
│   ├── extract_text.py             ← Step 2: chunk all notes/QP PDFs, tag by subject+module
│   ├── build_index.py              ← Step 3: embed chunks, build FAISS index
│   ├── generate_training_data.py   ← Step 4: generate fine-tuning dataset via GPT API
│   └── validate_dataset.py         ← Step 5: validate and clean the dataset
│
├── syllabuses/                     ← output of extract_syllabus.py (one .txt per subject)
│   ├── UCEST105_python.txt
│   ├── UCEST102_electrical.txt
│   └── ...
│
├── data/
│   ├── raw/
│   │   ├── notes/                  ← subject notes PDFs
│   │   ├── previous_qps/           ← past question paper PDFs
│   │   └── syllabus/               ← syllabus PDFs (input for extract_syllabus.py)
│   ├── processed/                  ← auto-created by extract_text.py
│   │   └── chunks.pkl
│   └── training/                   ← auto-created by generate_training_data.py
│       ├── raw_generated.jsonl     ← output of generation step
│       └── training_data.jsonl     ← output of validation step — upload this to Azure
│
├── faiss/                          ← auto-created by build_index.py
│   ├── ktu_index.faiss
│   └── ktu_index_meta.pkl
│
├── .env                            ← Azure credentials (never commit this)
└── requirements.txt
```

---

## PDF Naming Convention

All PDFs in `data/raw/` must follow this naming pattern. The keyword in the filename is how every script identifies which subject a file belongs to.

| File type | Pattern | Example |
|---|---|---|
| Syllabus | `<keyword>_syllabus.pdf` | `python_syllabus.pdf` |
| Notes (full subject) | `<keyword>_notes.pdf` | `chemistry_notes.pdf` |
| Notes (per module) | `<keyword>_mod1.pdf` | `physics_mod2.pdf` |
| Past question paper | `<keyword>_qp_<anything>.pdf` | `prog_c_qp_2024.pdf` |

**Keywords:** `python`, `electrical`, `chemistry`, `physics`, `prog_c`, `foundations`, `entrepreneur`

**Rules:**
- Past QPs **must** contain `_qp_` in the filename — this is how the system auto-detects them and adds them to training data at no extra API cost
- Module-specific notes (`_mod1.pdf`, `_mod2.pdf` etc.) are strongly preferred over one combined file — module number is read directly from the filename (100% reliable) rather than inferred from content keywords
- PDFs in subfolders work fine — all scripts use `os.walk()` so `data/raw/notes/python_mod1.pdf` is treated the same as `data/raw/python_mod1.pdf`
- If you only have handwritten/scanned notes for a subject, don't put them in `data/raw/`. The system still works using the syllabus txt alone — the model already knows standard content from its base training

---

## Environment Setup

**Install Python dependencies:**
```bash
pip install flask flask-cors openai python-dotenv pymupdf \
            faiss-cpu sentence-transformers torch numpy
```

**Create `.env` in the project root:**
```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=ktu-qp-finetuned
AZURE_OPENAI_BASE_DEPLOYMENT=gpt-4o-mini
```

`AZURE_OPENAI_BASE_DEPLOYMENT` is only used by `generate_training_data.py` to generate synthetic papers. `AZURE_OPENAI_DEPLOYMENT` is your fine-tuned model used by `app.py` at runtime.

**Always run scripts from the project root:**
```bash
# Correct — from ktu-qp-generator/
python scripts/extract_text.py

# Wrong — all relative paths (data/, faiss/, syllabuses/) will break
cd scripts && python extract_text.py
```

---

## Full Pipeline: First-Time Setup

### Step 1 — Extract syllabus module maps

Run once per subject. Reads the KTU syllabus PDF and produces a structured `.txt` file that maps each topic to its module. This file is the source of truth for all module boundary enforcement throughout the system.

```bash
python scripts/extract_syllabus.py \
  --pdf data/raw/syllabus/python_syllabus.pdf \
  --course_code UCEST105 \
  --subject Python \
  --output syllabuses/UCEST105_python.txt
```

Run for every subject:
```bash
python scripts/extract_syllabus.py --pdf data/raw/syllabus/electrical_syllabus.pdf  --course_code UCEST102 --subject "Electrical and Electronics"         --output syllabuses/UCEST102_electrical.txt
python scripts/extract_syllabus.py --pdf data/raw/syllabus/chemistry_syllabus.pdf   --course_code UCEST104 --subject Chemistry                            --output syllabuses/UCEST104_chemistry.txt
python scripts/extract_syllabus.py --pdf data/raw/syllabus/physics_syllabus.pdf     --course_code UCEST106 --subject Physics                              --output syllabuses/UCEST106_physics.txt
python scripts/extract_syllabus.py --pdf data/raw/syllabus/prog_c_syllabus.pdf      --course_code UCEST108 --subject "Programming in C"                   --output syllabuses/UCEST108_prog_c.txt
python scripts/extract_syllabus.py --pdf data/raw/syllabus/foundations_syllabus.pdf --course_code UCEST109 --subject "Foundations of Computing"           --output syllabuses/UCEST109_foundations.txt
python scripts/extract_syllabus.py --pdf data/raw/syllabus/entrepreneur_syllabus.pdf --course_code UCEST110 --subject "Engineering Entrepreneurship and IPR" --output syllabuses/UCEST110_entrepreneur.txt
```

**After each run: open the output `.txt` and verify it manually.** The file should look like:

```
COURSE CODE: UCEST105
SUBJECT: Python
============================================================

MODULE 1: Module 1
--------------------------------------------------
  - Problem-solving strategies: definition, Trial and Error...
  - Variables, numeric and string data types
  ...

MODULE 2: Module 2
--------------------------------------------------
  - Pseudocode: definition, reasons for use
  ...
```

Verify that:
- Exactly 4 modules are present (Module 1 through Module 4)
- Topics are in the right module
- No administrative text slipped in (marks, credits, contact hours etc.)

If you see `No module headers found`, the syllabus PDF likely uses Roman numerals (`Module I`, `Module II`) or `Unit 1` instead of `Module 1`. Open the PDF, check the exact header format, and edit the regex in `extract_syllabus.py` to match.

This is the most important manual step in the entire pipeline. If module boundaries are wrong here, the model will learn wrong boundaries and no downstream fix can correct it.

### Step 2 — Chunk and index all PDFs

Reads every PDF in `data/raw/` (all subfolders), splits into 300-word overlapping chunks, and tags each chunk with subject and module number.

```bash
python scripts/extract_text.py
```

Module assignment uses two tiers in priority order:
1. **Filename tag** (`python_mod2.pdf` → Module 2) — used when available, always correct
2. **Keyword scoring** against the syllabus txt — fallback for untagged files like `python_notes.pdf`

The script prints per-file stats:
```
Processing: data/raw/notes/python_mod1.pdf
  Subject: Python | Type: notes | Chunks: 42 | filename-tagged: 42 | keyword-tagged: 0 | untagged: 0

Processing: data/raw/notes/python_notes.pdf
  Subject: Python | Type: notes | Chunks: 120 | filename-tagged: 0 | keyword-tagged: 98 | untagged: 22
```

If a subject shows `untagged: 140 | keyword-tagged: 0`, the syllabus txt was not found for that subject. Check that the txt exists in `syllabuses/` and its filename contains the correct keyword.

Output: `data/processed/chunks.pkl`

### Step 3 — Build FAISS vector index

Embeds all chunks using `all-MiniLM-L6-v2` (runs locally, free, no API calls) and builds the similarity search index.

```bash
python scripts/build_index.py
```

Takes a few minutes depending on chunk count. Re-run this whenever you add new PDFs to `data/raw/`.

Output:
- `faiss/ktu_index.faiss`
- `faiss/ktu_index_meta.pkl`

### Step 4 — Generate fine-tuning dataset

Before running, open `generate_training_data.py` and configure two things:

**1. Number of samples per subject** (line 35):
```python
SAMPLES_PER_SUBJECT = 10
```

| Samples | Cost (7 subjects) | Recommended when |
|---|---|---|
| 5 | ~₹5 | You have 3+ real QPs per subject |
| 10 | ~₹10 | You have 1–2 real QPs (default) |
| 20 | ~₹20 | You have no real QPs |

**2. Uncomment all subjects you have syllabus files for** (lines 37–45):
```python
SUBJECTS = [
    ("Python",                               "1 & 2", "UCEST105_python.txt"),
    ("Electrical and Electronics",           "1 & 2", "UCEST102_electrical.txt"),
    ("Chemistry",                            "1 & 2", "UCEST104_chemistry.txt"),
    ("Physics",                              "1 & 2", "UCEST106_physics.txt"),
    ("Programming in C",                     "1 & 2", "UCEST108_prog_c.txt"),
    ("Foundations of Computing",             "1 & 2", "UCEST109_foundations.txt"),
    ("Engineering Entrepreneurship and IPR", "1 & 2", "UCEST110_entrepreneur.txt"),
]
```

Then run:
```bash
python scripts/generate_training_data.py
```

What this does:
1. Loads the FAISS index
2. Loads each subject's syllabus txt and builds its system prompt (full module map injected)
3. Scans `data/raw/` for past QP PDFs (files containing `_qp_` in name) and adds them as training entries — **no extra API cost**
4. Generates `SAMPLES_PER_SUBJECT` synthetic papers per subject, each with a different style variation (practical, theoretical, industry scenario, numerical etc.)
5. Saves all entries to `data/training/raw_generated.jsonl`

**This step costs money** — it calls the Azure OpenAI API for each synthetic paper. Every other step in the pipeline is free.

Output: `data/training/raw_generated.jsonl`

### Step 5 — Validate and clean the dataset

```bash
python scripts/validate_dataset.py
```

Reads `raw_generated.jsonl` and applies different validation rules depending on entry type:

**Synthetic papers must pass all of:**
- Contains PART A and PART B headers
- At least 4 OR pairs (one per module)
- At least 14 numbered questions
- Has (a)/(b) subparts in Part B
- At least 8 mark labels in format `(3)`, `(5)` etc.
- No subpart carries fewer than 3 marks
- All 4 modules referenced
- At least 300 words total

**Real past QP entries** are checked only for minimum content (100+ words). Raw PDF text has page numbers, CO labels, and headers that would fail structural checks — so structural validation is skipped for these.

The script prints a full report:
```
VALIDATION REPORT
==================================================
Total samples:    76
Valid samples:    68  (real QPs: 6, synthetic: 62)
Invalid samples:  8
Pass rate:        89.5%
```

Target pass rate: 80%+. If lower, the most common causes are:
- Syllabus txt has too few topics — check each `syllabuses/*.txt`
- FAISS index is empty or nearly empty for that subject — check Step 2 output
- GPT returning truncated responses — increase `max_tokens` in `generate_question_paper()` in `generate_training_data.py`

Output: `data/training/training_data.jsonl` — upload this to Azure.

---

## Fine-Tuning on Azure

1. Go to **Azure OpenAI Studio → Fine-tuning → Create new fine-tuning job**
2. Upload `data/training/training_data.jsonl` as the training file
3. Select base model: `gpt-4o-mini` (recommended) or `gpt-35-turbo`
4. Start the job — takes 30–90 minutes
5. Once complete, go to **Deployments → Create**
6. Select your fine-tuned model from the dropdown
7. Give it a deployment name (e.g. `ktu-qp-finetuned`)
8. Update `AZURE_OPENAI_DEPLOYMENT` in `.env` to match exactly — it is case-sensitive

**Cost notes:**
| Activity | Cost |
|---|---|
| Fine-tuning job | ~₹500 one-time |
| Deployment while active | ~₹500/day (charged even at zero requests) |
| Undeployed model (weights saved) | ₹0/day |
| Re-deploying | ~5 minutes, no extra charge |

Recommended workflow: deploy only when actively testing or serving requests → undeploy when done. Azure auto-deletes deployments after 15 days of inactivity, but the model weights remain saved and can be redeployed any time.

---

## Running the API

```bash
python backend/app.py
```

Server starts at `http://localhost:5000`. The FAISS index and embedding model load once at startup — this takes about 10 seconds. Every subsequent request is fast.

---

## API Reference

### `POST /generate`

Generate a KTU question paper.

**Request body:**
```json
{
  "subject": "Python",
  "semester": "1 & 2"
}
```

**Success response `200`:**
```json
{
  "question_paper": "PART A\n(Answer all questions. Each question carries 3 marks)\n\n1. ...",
  "latex": false
}
```

`latex` is `true` for Physics, Chemistry, and Electrical and Electronics. When `true`, the frontend must render the text through MathJax or KaTeX — otherwise equations like `$x^2 + 3x = 0` will appear as raw text.

**Error responses:**
```json
{ "error": "Subject is required." }                        // 400
{ "error": "Subject 'X' is not supported." }              // 400
{ "error": "Semester must be '1 & 2'." }                  // 400
{ "error": "Retrieval failed: ..." }                       // 500
{ "error": "Model inference failed: ..." }                 // 500
{ "error": "Model returned an empty response." }           // 500
```

### `GET /subjects`

Returns all supported subjects and which ones have syllabus files loaded.

**Response:**
```json
{
  "subjects": [
    "Python",
    "Electrical and Electronics",
    "Chemistry",
    "Physics",
    "Programming in C",
    "Foundations of Computing",
    "Engineering Entrepreneurship and IPR"
  ],
  "semesters": ["1 & 2"],
  "syllabus_loaded": {
    "Python": true,
    "Electrical and Electronics": true,
    "Chemistry": false,
    ...
  }
}
```

`syllabus_loaded: false` for a subject means it has no syllabus txt file. The subject will still generate papers, but without module boundary enforcement — module-topic mismatches become more likely.

### `GET /health`

```json
{ "status": "ok" }
```

---

## Frontend Integration

### Basic fetch

```javascript
const response = await fetch('http://localhost:5000/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ subject: 'Physics', semester: '1 & 2' })
})
const { question_paper, latex } = await response.json()
```

### MathJax rendering (for Physics, Chemistry, Electrical and Electronics)

Add to your HTML:
```html
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
```

In your JavaScript, after setting the paper content:
```javascript
const outputDiv = document.getElementById('paper-output')
outputDiv.textContent = question_paper   // set as text, not innerHTML

if (latex) {
  MathJax.typesetPromise([outputDiv])    // render LaTeX in that element
}
```

---

## Manual Quality Validation

After generating test papers, check a sample manually. This is the most reliable way to catch module-topic mismatches that automated checks cannot detect.

**Part A checklist:**
- Q1 and Q2 use only Module 1 topics
- Q3 and Q4 use only Module 2 topics
- Q5 and Q6 use only Module 3 topics
- Q7 and Q8 use only Module 4 topics
- Each question ends with `(3)`, no other mark values in Part A
- No "Module 1", "Module 2" labels appear in Part A (numbers only)

**Part B checklist:**
- "Module 1" label before Q9–10, all questions use only Module 1 topics
- "Module 2" label before Q11–12, all questions use only Module 2 topics
- "Module 3" label before Q13–14, all questions use only Module 3 topics
- "Module 4" label before Q15–16, all questions use only Module 4 topics
- OR appears on its own line between every pair
- Subpart splits are (5)+(4) or (6)+(3), totalling 9
- Single-question format `(9)` used at most twice per paper

If module-topic mismatches are found, the root cause is almost always incorrect module boundaries in the syllabus `.txt` file. Fix the txt, re-run from Step 4, and retrain.

---

## Adding a New Subject

1. Add syllabus PDF to `data/raw/syllabus/`
2. Add notes and past QPs to `data/raw/` following the naming convention
3. Run `extract_syllabus.py` and verify the output txt manually
4. Re-run `extract_text.py` and `build_index.py`
5. Add the subject to `SUBJECTS` in `generate_training_data.py`
6. Add the subject to `SUPPORTED_SUBJECTS` and `SUBJECT_KEYWORD_MAP` in `app.py`
7. Add the keyword to `SUBJECT_MAP` and `SUBJECT_KEYWORD_MAP` in `extract_text.py`
8. Re-run `generate_training_data.py` and `validate_dataset.py`
9. Re-fine-tune with the updated dataset on Azure

---

## Common Issues and Fixes

| Symptom | Cause | Fix |
|---|---|---|
| `No module headers found` in extract_syllabus | PDF uses Roman numerals (`Module I`) or `Unit 1` | Check the PDF, edit the regex in `extract_syllabus.py` |
| `No text extracted` warning | PDF is image-based (scanned) | Run OCR first, or skip that file |
| All chunks `untagged`, `keyword-tagged: 0` | Syllabus txt missing for that subject | Check `syllabuses/` — filename must contain the subject keyword |
| High rejection rate in validate_dataset | Papers too short or structurally incomplete | Check FAISS index has chunks for that subject; consider raising `SAMPLES_PER_SUBJECT` |
| Wrong subject detected for a PDF | Filename doesn't follow the convention | Rename using `<keyword>_type.pdf` |
| `FileNotFoundError` on syllabus in generate_training_data | Syllabus txt filename doesn't match `SUBJECTS` list | Check exact filename in `syllabuses/` and update `SUBJECTS` |
| Model returns empty response | Deployment name mismatch | Check `AZURE_OPENAI_DEPLOYMENT` in `.env` matches Azure exactly (case-sensitive) |
| Equations render as plain `$x^2$` text | MathJax not called after DOM update | Call `MathJax.typesetPromise()` after setting paper content |
| Fine-tuning file rejected by Azure | Structural issue in JSONL | Re-run `validate_dataset.py`; check all entries have 3 messages with correct roles |
| Module-topic mismatches in generated papers | Wrong module boundaries in syllabus txt | Fix the txt file, re-generate training data, retrain |
