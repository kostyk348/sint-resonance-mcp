import math
import re
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "shall", "should", "may", "might", "must", "i", "you", "he", "she", "it",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "this", "that", "these", "those", "and", "or", "but", "if",
    "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "also", "now",
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def compute_tf(text_tokens: list[str]) -> dict[str, float]:
    if not text_tokens:
        return {}
    counts = Counter(text_tokens)
    total = len(text_tokens)
    return {word: count / total for word, count in counts.items()}


class TfidfIndex:
    def __init__(self):
        self.documents: list[dict] = []
        self.idf: dict[str, float] = {}
        self.dirty = True

    def add(self, doc_id: str, text: str):
        tokens = tokenize(text)
        self.documents.append({"id": doc_id, "text": text, "tokens": tokens})
        self.dirty = True

    def remove(self, doc_id: str):
        self.documents = [d for d in self.documents if d["id"] != doc_id]
        self.dirty = True

    def rebuild(self):
        n = len(self.documents)
        if n == 0:
            self.idf = {}
            return
        doc_freq: Counter = Counter()
        for doc in self.documents:
            unique = set(doc["tokens"])
            for word in unique:
                doc_freq[word] += 1
        self.idf = {
            word: math.log((n + 1) / (freq + 1)) + 1
            for word, freq in doc_freq.items()
        }
        self.dirty = False

    def _ensure_built(self):
        if self.dirty:
            self.rebuild()

    def vectorize(self, text: str) -> dict[str, float]:
        self._ensure_built()
        tokens = tokenize(text)
        tf = compute_tf(tokens)
        vec = {}
        for word, tf_val in tf.items():
            if word in self.idf:
                vec[word] = tf_val * self.idf[word]
            else:
                vec[word] = tf_val
        return vec

    def cosine_similarity(self, vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = 0.0
        for word, val in vec_a.items():
            if word in vec_b:
                dot += val * vec_b[word]
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query: str, k: int = 5) -> list[dict]:
        self._ensure_built()
        q_vec = self.vectorize(query)
        scores = []
        for doc in self.documents:
            d_vec = self.vectorize(doc["text"])
            sim = self.cosine_similarity(q_vec, d_vec)
            if sim > 0:
                scores.append({"id": doc["id"], "similarity": round(sim, 4), "text": doc["text"]})
        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:k]


def keyword_overlap(text_a: str, text_b: str) -> float:
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)
