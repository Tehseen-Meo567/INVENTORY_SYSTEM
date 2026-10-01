"""Reads vendor documents and finds the most relevant passages for a vendor.
Uses ChromaDB if installed and working, otherwise a built-in TF-IDF search (same idea, no downloads)."""
import os, glob
from pathlib import Path

class Retriever:
    def __init__(self, docs_dir):
        self.chunks = []   # dicts: text, vendor_id, source
        for f in sorted(glob.glob(str(Path(docs_dir) / "*.txt"))):
            vid = Path(f).name.split("_")[0]
            for para in [x.strip() for x in open(f, encoding="utf-8").read().split("\n\n") if x.strip()]:
                self.chunks.append({"text": para, "vendor_id": vid, "source": Path(f).name})
        self.mode = "tfidf"
        if os.getenv("USE_CHROMA", "auto") != "0":
            try:
                import chromadb
                self._col = chromadb.EphemeralClient().get_or_create_collection("vendor_docs")
                self._col.add(ids=[str(i) for i in range(len(self.chunks))],
                              documents=[c["text"] for c in self.chunks],
                              metadatas=[{"vendor_id": c["vendor_id"], "source": c["source"]} for c in self.chunks])
                self._col.query(query_texts=["test"], n_results=1)   # probe (downloads embedding model first time)
                self.mode = "chromadb"
            except Exception as e:
                print("ChromaDB not used, falling back to TF-IDF:", type(e).__name__)
        if self.mode == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(stop_words="english")
            self._mat = self._vec.fit_transform([c["text"] for c in self.chunks])

    def search(self, query, vendor_id, k=6):
        if self.mode == "chromadb":
            r = self._col.query(query_texts=[query], n_results=k, where={"vendor_id": vendor_id})
            return [{"text": d, "source": m["source"]} for d, m in zip(r["documents"][0], r["metadatas"][0])]
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(self._vec.transform([query]), self._mat).ravel()
        idx = [i for i in sims.argsort()[::-1] if self.chunks[i]["vendor_id"] == vendor_id][:k]
        return [{"text": self.chunks[i]["text"], "source": self.chunks[i]["source"]} for i in idx]
