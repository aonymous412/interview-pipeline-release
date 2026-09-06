"""C2 -- Semantic Clusterer. K-Means sur les embeddings des descriptions,
aucun fine-tuning. (sentence-transformers -- modele all-MiniLM-L6-v2)

NOTE : all-MiniLM-L6-v2 se telecharge depuis huggingface.co au premier import
de SentenceTransformer -- bloque dans ce sandbox (meme restriction reseau que
les checkpoints C1/C4). A executer sur l'agent local.
"""
from __future__ import annotations


class C2Clusterer:
    def __init__(self):
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def cluster(self, params: list[dict]) -> list[list[dict]]:
        if len(params) <= 2:
            return [params] if params else []

        from sklearn.cluster import KMeans

        embedder = self._get_embedder()
        embeddings = embedder.encode([p["description"] for p in params])
        k = max(1, len(params) // 3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)

        clusters: list[list[dict]] = [[] for _ in range(k)]
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(params[i])
        clusters = [c for c in clusters if c]
        return sorted(clusters, key=lambda c: max(p["weight"] for p in c), reverse=True)
