from .index import get_index


def suggest_for_context(context_snapshot: str, k: int = 5) -> list[dict]:
    idx = get_index()
    return idx.proactive(context_snapshot, k)
