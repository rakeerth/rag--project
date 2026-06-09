from ddgs import DDGS
from ddgs.exceptions import TimeoutException


def web_search(query: str, max_results: int = 5) -> str:
    """Search the live web and return text results for a query."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except TimeoutException:
        return "Web search timed out. Please try again."
    except Exception as e:
        return f"Web search failed: {e}"

    if not results:
        return "No web results found."
    formatted = []
    for r in results:
        formatted.append(
            f"{r.get('title', '')}\n{r.get('body', '')}\n({r.get('href', '')})")
    return "\n\n".join(formatted)
