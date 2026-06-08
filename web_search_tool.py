from ddgs import DDGS

# A web search tool the agent can call.
# Swapping to Brave later means replacing only what's inside this file;
# the rest of the system just calls `web_search(query)`.


def web_search(query: str, max_results: int = 5) -> str:
    """Search the live web and return text results for a query."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    if not results:
        return "No web results found."
    # format the results into plain text for the LLM
    formatted = []
    for r in results:
        formatted.append(
            f"{r.get('title', '')}\n{r.get('body', '')}\n({r.get('href', '')})")
    return "\n\n".join(formatted)


# --- Standalone test: run this file directly to prove the tool works ---
if __name__ == "__main__":
    print("Testing web search tool...\n")
    test_query = "current weather in London"
    print(f"Query: {test_query}\n")
    print("Result:\n", web_search(test_query))
