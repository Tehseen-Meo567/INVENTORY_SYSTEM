"""Document search (RAG) over vendor documents.

Engines, tried in this order:
  1. ChromaDB with its normal embedding model (needs internet the first time to download a small model)
  2. ChromaDB with a built-in offline "hash" embedding (works without internet, less smart)
  3. TF-IDF word matching (no ChromaDB at all)
Set USE_CHROMA=0 to force 3, or USE_CHROMA=hash to force 2. The sidebar shows which engine is active."""
import os, re, glob, zlib
from pathlib import Path
import numpy as np

class HashEmbedding:
    """Tiny offline embedding: counts words into 384 buckets. Good enough to demo ChromaDB without downloading a model."""
    DIM = 384
    def __init__(self): pass
    def __call__(self, input):
        out = []
        for t in input:
            v = np.zeros(self.DIM)
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[zlib.crc32(w.encode()) % self.DIM] += 1
            n = np.linalg.norm(v) or 1.0
            out.append((v / n).tolist())
        return out
    @staticmethod
    def name(): return "supplysense_hash_embedding"
    def get_config(self): return {}
    @staticmethod
    def build_from_config(config): return HashEmbedding()
    def embed_query(self, input): return self.__call__(input)
    def is_legacy(self): return False
    def default_space(self): return "cosine"
    def supported_spaces(self): return ["cosine", "l2", "ip"]

class Retriever:
    def __init__(self, docs_dir, db_path=None, engine=None):
        self.chunks = []   # dicts: text, vendor_id, source
        for f in sorted(glob.glob(str(Path(docs_dir) / "*.txt"))):
            vid = Path(f).name.split("_")[0]
            for para in [x.strip() for x in open(f, encoding="utf-8").read().split("\n\n") if x.strip()]:
                self.chunks.append({"text": para, "vendor_id": vid, "source": Path(f).name})
        self.mode, self.note = "tfidf", ""
        self.db_path = str(db_path or Path(docs_dir).parent.parent / "chroma_db")
        pref = {"tfidf": "0", "hash": "hash", "default": "auto"}.get(engine) or os.getenv("USE_CHROMA", "auto")
        if pref != "0":
            order = ["hash"] if pref == "hash" else ["default", "hash"]
            for emb in order:
                try:
                    self._init_chroma(emb); break
                except Exception as e:
                    self.note = f"ChromaDB ({emb} embeddings) failed: {type(e).__name__}"
                    print(self.note)
        if self.mode == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(stop_words="english")
            self._mat = self._vec.fit_transform([c["text"] for c in self.chunks])

    def _init_chroma(self, emb):
        import chromadb
        client = chromadb.PersistentClient(path=self.db_path)
        try: client.delete_collection("vendor_docs")
        except Exception: pass
        kw = {"embedding_function": HashEmbedding()} if emb == "hash" else {}
        col = client.create_collection("vendor_docs", metadata={"hnsw:space": "cosine"}, **kw)
        col.add(ids=[str(i) for i in range(len(self.chunks))], documents=[c["text"] for c in self.chunks],
                metadatas=[{"vendor_id": c["vendor_id"], "source": c["source"]} for c in self.chunks])
        col.query(query_texts=["probe"], n_results=1)      # triggers model download for 'default'
        self._col = col
        self.mode = "chromadb" if emb == "default" else "chromadb (offline hash embeddings)"

    def info(self):
        return {"engine": self.mode, "chunks": len(self.chunks), "vendors": len({c["vendor_id"] for c in self.chunks}),
                "folder": self.db_path if self.mode.startswith("chromadb") else "(none, in memory)", "note": self.note}

    def search(self, query, vendor_id=None, k=6):
        """Returns [{text, source, vendor_id, score}] best first. score: 0-1, higher = more similar."""
        if self.mode.startswith("chromadb"):
            kw = {"where": {"vendor_id": vendor_id}} if vendor_id else {}
            r = self._col.query(query_texts=[query], n_results=k, **kw)
            return [{"text": d, "source": m["source"], "vendor_id": m["vendor_id"], "score": round(1 - dist, 3)}
                    for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(self._vec.transform([query]), self._mat).ravel()
        idx = [i for i in sims.argsort()[::-1] if vendor_id is None or self.chunks[i]["vendor_id"] == vendor_id][:k]
        return [{"text": self.chunks[i]["text"], "source": self.chunks[i]["source"], "vendor_id": self.chunks[i]["vendor_id"],
                 "score": round(float(sims[i]), 3)} for i in idx]
