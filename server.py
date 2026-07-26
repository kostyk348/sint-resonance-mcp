from fastmcp import FastMCP
from config import NAME, VERSION
from resonance.index import get_index
from resonance.proactive import suggest_for_context

mcp = FastMCP(NAME, version=VERSION)
idx = get_index()


@mcp.tool()
def resonance_index(domain: str, problem: str, solution: str, tags: list[str]) -> dict:
    entry_id = idx.index(domain, problem, solution, tags)
    return {"status": "indexed", "id": entry_id, "domain": domain, "problem": problem}


@mcp.tool()
def resonance_find(query: str, k: int = 5, domains: list[str] | None = None) -> list[dict]:
    return idx.find(query, k, domains)


@mcp.tool()
def resonance_structural(problem_structure: str, k: int = 5) -> list[dict]:
    return idx.find_structural(problem_structure, k)


@mcp.tool()
def resonance_proactive(context_snapshot: str, k: int = 5) -> list[dict]:
    return suggest_for_context(context_snapshot, k)


@mcp.tool()
def resonance_domains() -> list[str]:
    return idx.domains()


@mcp.tool()
def resonance_stats() -> dict:
    return idx.stats()
