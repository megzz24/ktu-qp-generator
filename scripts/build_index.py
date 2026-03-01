import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------------------------
# CONFIG
# -----------------------------------------------

CHUNKS_PATH = "data/processed/chunks.pkl"
INDEX_PATH = "faiss/ktu_index.faiss"
META_PATH = "faiss/ktu_index_meta.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"  # runs locally, no API needed

# -----------------------------------------------
# MAIN
# -----------------------------------------------


def build_faiss_index() -> None:
    os.makedirs("faiss", exist_ok=True)

    # Load chunks
    print("Loading chunks...")
    with open(CHUNKS_PATH, "rb") as f:
        chunks, metadata = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks.")

    # Load embedding model
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Generate embeddings
    print("Generating embeddings (this may take a few minutes)...")
    embeddings = model.encode(
        chunks, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    embeddings = embeddings.astype("float32")

    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # type: ignore

    print(f"Index built with {index.ntotal} vectors of dimension {dimension}.")

    # Save index and metadata
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}")

    with open(META_PATH, "wb") as f:
        pickle.dump((chunks, metadata), f)
    print(f"Metadata saved to {META_PATH}")


if __name__ == "__main__":
    build_faiss_index()
