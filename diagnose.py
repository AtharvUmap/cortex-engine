"""Diagnose where the resume ranks in vector retrieval. Delete after use."""
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

q = "What is Atharv Umap experience"

# Connect to the live Chroma collection directly (not via ParentDocumentRetriever)
# so we can look at raw child-vector ranking before parent-dedup.
emb = OllamaEmbeddings(model="nomic-embed-text")
vs = Chroma(
    collection_name="child_chunks",
    embedding_function=emb,
    persist_directory="./db",
)

print(f"Total children in index: {vs._collection.count()}")
print()
print(f"=== TOP 30 CHILDREN FOR: {q!r} ===")
results = vs.similarity_search_with_score(q, k=30)
for i, (doc, score) in enumerate(results):
    src = Path(doc.metadata.get("source", "?")).name
    preview = doc.page_content[:80].replace("\n", " ")
    print(f"{i+1:2d}. score={score:.3f}  src={src}")
    print(f"     {preview}")

print()
print("=== SOURCE COUNTS IN TOP 30 ===")
from collections import Counter
sources = Counter(Path(d.metadata.get("source", "?")).name for d, _ in results)
for src, count in sources.most_common():
    print(f"  {count:3d}  {src}")
