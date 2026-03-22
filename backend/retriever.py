import faiss
import pickle
from sentence_transformers import SentenceTransformer

# -----------------------------------------------
# CONFIG
# -----------------------------------------------

import os
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "faiss", "ktu_index.faiss")
META_PATH  = os.path.join(BASE_DIR, "faiss", "ktu_index_meta.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"

# -----------------------------------------------
# LOAD ONCE AT MODULE LEVEL
# (so Flask doesn't reload on every request)
# -----------------------------------------------

embedder = SentenceTransformer(MODEL_NAME)

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    chunks, metadata = pickle.load(f)

# -----------------------------------------------
# RETRIEVAL
# -----------------------------------------------


def retrieve_context(subject: str, top_k: int = 8) -> str:
    """
    Retrieve relevant chunks from the FAISS index for a given subject.
    Returns balanced context across all 4 modules.
    """
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
