import hashlib
import re


PROBLEM_TYPES = {
    "performance": ["slow", "latency", "throughput", "bottleneck", "overhead", "cache miss"],
    "memory": ["leak", "oom", "buffer", "allocation", "fragmentation", "stack", "heap"],
    "concurrency": ["race", "deadlock", "lock", "thread", "mutex", "semaphore", "atomic"],
    "security": ["injection", "overflow", "xss", "csrf", "auth", "permission", "crypto"],
    "networking": ["timeout", "retry", "packet loss", "connection", "protocol", "socket"],
    "data": ["consistency", "corruption", "duplicate", "schema", "migration", "query"],
    "build": ["compile", "link", "dependency", "version", "toolchain", "cross compile"],
    "deploy": ["rollout", "rollback", "config", "env", "container", "orchestrate"],
}

APPROACH_PATTERNS = {
    "lazy": ["lazy", "defer", "on demand", "just in time", "lazy init"],
    "caching": ["cache", "memoize", "buffer pool", "reuse", "precompute"],
    "batching": ["batch", "coalesce", "aggregate", "bulk", "chunk"],
    "backoff": ["backoff", "retry", "exponential", "timeout", "circuit breaker"],
    "pooling": ["pool", "recycle", "object pool", "connection pool", "arena"],
    "sharding": ["shard", "partition", "segment", "striping", "consistent hash"],
    "async": ["async", "non-blocking", "callback", "event loop", "promise", "future"],
    "lock_free": ["lock free", "lockfree", "wait free", "cas", "atomic", "rcu"],
    "divide_conquer": ["divide", "conquer", "recursive", "merge", "split", "map reduce"],
}

ENTROPY_BOOST_WORDS = {
    "unexpected", "surprising", "weird", "strange", "unusual",
    "bug", "crash", "corrupt", "wrong", "incorrect",
}


def extract_problem_type(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for ptype, keywords in PROBLEM_TYPES.items():
        if any(kw in text_lower for kw in keywords):
            found.append(ptype)
    return found


def extract_approach_pattern(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for apattern, keywords in APPROACH_PATTERNS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(apattern)
    return found


def compute_entropy_boost(text: str) -> float:
    text_lower = text.lower()
    boost = 0.0
    for word in ENTROPY_BOOST_WORDS:
        if word in text_lower:
            boost += 0.05
    return min(boost, 0.3)


def extract_key_terms(text: str, limit: int = 10) -> list[str]:
    text_lower = text.lower()
    words = re.findall(r"[a-z][a-z0-9_]{2,}", text_lower)
    from collections import Counter
    freq = Counter(words)
    return [w for w, _ in freq.most_common(limit)]


def make_fingerprint(problem: str, domain: str, tags: list[str]) -> str:
    problem_types = extract_problem_type(problem)
    approaches = extract_approach_pattern(problem)
    key_terms = extract_key_terms(problem, limit=8)
    tag_set = sorted(t.lower().replace(" ", "_") for t in tags)
    raw = "|".join([
        domain.lower(),
        ":".join(sorted(problem_types)),
        ":".join(sorted(approaches)),
        ":".join(sorted(key_terms)),
        ":".join(tag_set),
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def structural_similarity(
    problem_a: str, domain_a: str, tags_a: list[str],
    problem_b: str, domain_b: str, tags_b: list[str],
) -> float:
    fp_a = make_fingerprint(problem_a, domain_a, tags_a)
    fp_b = make_fingerprint(problem_b, domain_b, tags_b)

    score = 0.0

    if fp_a == fp_b:
        score += 0.6

    types_a = set(extract_problem_type(problem_a))
    types_b = set(extract_problem_type(problem_b))
    if types_a and types_b:
        overlap = types_a & types_b
        score += 0.15 * (len(overlap) / max(len(types_a | types_b), 1))

    apps_a = set(extract_approach_pattern(problem_a))
    apps_b = set(extract_approach_pattern(problem_b))
    if apps_a and apps_b:
        overlap = apps_a & apps_b
        score += 0.15 * (len(overlap) / max(len(apps_a | apps_b), 1))

    if domain_a.lower() == domain_b.lower():
        score += 0.1

    return round(min(score, 1.0), 4)
