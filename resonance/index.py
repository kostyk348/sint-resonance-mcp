import json
import os
import threading
from pathlib import Path
from .embeddings import TfidfIndex, keyword_overlap
from .structural import structural_similarity, make_fingerprint, extract_problem_type, extract_approach_pattern

DATA_DIR = Path.home() / ".config" / "opencode" / "resonance"
INDEX_FILE = DATA_DIR / "index.json"


class ResonanceIndex:
    def __init__(self):
        self.entries: dict[str, dict] = {}
        self.tfidf = TfidfIndex()
        self.lock = threading.Lock()
        self._seq = 0
        self._load()

    def _next_id(self) -> str:
        self._seq += 1
        return f"res:{self._seq:04d}"

    def _load(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if INDEX_FILE.exists():
            try:
                data = json.loads(INDEX_FILE.read_text())
                self.entries = data.get("entries", {})
                self._seq = data.get("seq", len(self.entries))
                for eid, entry in self.entries.items():
                    text = f"{entry['problem']} {entry['solution']} {' '.join(entry['tags'])}"
                    self.tfidf.add(eid, text)
            except (json.JSONDecodeError, KeyError):
                self.entries = {}
                self._seq = 0

    def _save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": self.entries,
            "seq": self._seq,
        }
        INDEX_FILE.write_text(json.dumps(data, indent=2))

    def index(self, domain: str, problem: str, solution: str, tags: list[str]) -> str:
        tags = [t.strip().lower() for t in tags if t.strip()]
        fingerprint = make_fingerprint(problem, domain, tags)
        entry_id = self._next_id()
        entry = {
            "id": entry_id,
            "domain": domain,
            "problem": problem,
            "solution": solution,
            "tags": tags,
            "fingerprint": fingerprint,
            "problem_types": extract_problem_type(problem),
            "approach_patterns": extract_approach_pattern(problem),
        }
        with self.lock:
            self.entries[entry_id] = entry
            text = f"{problem} {solution} {' '.join(tags)}"
            self.tfidf.add(entry_id, text)
            self._save()
        return entry_id

    def find(self, query: str, k: int = 5, domains: list[str] | None = None) -> list[dict]:
        with self.lock:
            candidates = self._filter_domains(domains)
            results = self.tfidf.search(query, k * 2)

            scored = []
            seen = set()
            for r in results:
                if r["id"] not in candidates:
                    continue
                entry = self.entries[r["id"]]
                overlap = keyword_overlap(query, entry["problem"] + " " + entry["solution"])
                combined = (r["similarity"] * 0.6) + (overlap * 0.4)
                scored.append(self._format_resonance(entry, combined, "keyword+tfidf match"))
                seen.add(r["id"])

            for eid, entry in candidates.items():
                if eid in seen:
                    continue
                overlap = keyword_overlap(query, entry["problem"])
                if overlap > 0.1:
                    scored.append(self._format_resonance(entry, overlap * 0.5, "keyword overlap"))

            scored.sort(key=lambda x: x["similarity"], reverse=True)
            return scored[:k]

    def find_structural(
        self, problem_structure: str, k: int = 5,
        domain: str = "", tags: list[str] | None = None,
    ) -> list[dict]:
        tags = tags or []
        with self.lock:
            scored = []
            for entry in self.entries.values():
                sim = structural_similarity(
                    problem_structure, domain, tags,
                    entry["problem"], entry["domain"], entry["tags"],
                )
                if sim > 0.05:
                    scored.append(self._format_resonance(
                        entry, sim,
                        f"structural match (types: {entry['problem_types']}, approaches: {entry['approach_patterns']})",
                    ))
            scored.sort(key=lambda x: x["similarity"], reverse=True)
            return scored[:k]

    def proactive(self, context_snapshot: str, k: int = 5) -> list[dict]:
        with self.lock:
            if not self.entries:
                return []
            tfidf_results = self.tfidf.search(context_snapshot, k * 2)
            scored = []
            seen = set()

            for r in tfidf_results:
                entry = self.entries[r["id"]]
                overlap = keyword_overlap(context_snapshot, entry["problem"])
                combined = (r["similarity"] * 0.5) + (overlap * 0.3)
                entry_types = ", ".join(entry["problem_types"])
                entry_approaches = ", ".join(entry["approach_patterns"])
                reason = f"TF-IDF similarity {r['similarity']:.2f}, keyword overlap {overlap:.2f}"
                if entry_types:
                    reason += f", problem type: {entry_types}"
                if entry_approaches:
                    reason += f", approach: {entry_approaches}"
                scored.append(self._format_resonance(entry, combined, reason))
                seen.add(r["id"])

            for eid, entry in self.entries.items():
                if eid in seen:
                    continue
                overlap = keyword_overlap(context_snapshot, entry["problem"])
                if overlap > 0.15:
                    scored.append(self._format_resonance(entry, overlap * 0.4, "keyword overlap"))
                else:
                    str_sim = structural_similarity(
                        context_snapshot, "", [],
                        entry["problem"], entry["domain"], entry["tags"],
                    )
                    if str_sim > 0.2:
                        scored.append(self._format_resonance(entry, str_sim * 0.6, "structural hint"))

            scored.sort(key=lambda x: x["similarity"], reverse=True)
            return scored[:k]

    def domains(self) -> list[str]:
        with self.lock:
            return sorted(set(e["domain"] for e in self.entries.values()))

    def stats(self) -> dict:
        with self.lock:
            domain_counts = {}
            for e in self.entries.values():
                domain_counts[e["domain"]] = domain_counts.get(e["domain"], 0) + 1
            return {
                "total_entries": len(self.entries),
                "domains": len(domain_counts),
                "domain_breakdown": domain_counts,
            }

    def _filter_domains(self, domains: list[str] | None) -> dict:
        if not domains:
            return self.entries
        return {
            eid: entry for eid, entry in self.entries.items()
            if entry["domain"] in domains
        }

    def _format_resonance(self, entry: dict, similarity: float, reason: str) -> dict:
        return {
            "match_id": entry["id"],
            "domain": entry["domain"],
            "problem": entry["problem"],
            "similarity": round(min(similarity, 1.0), 4),
            "insight": entry["solution"],
            "relevance_reason": reason,
        }


_index = ResonanceIndex()


def get_index() -> ResonanceIndex:
    return _index
