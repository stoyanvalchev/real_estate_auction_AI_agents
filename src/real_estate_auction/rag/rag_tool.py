from crewai.tools import BaseTool
from real_estate_auction.rag.rag_pipeline import search


class PropertySearchTool(BaseTool):
    name: str = "property_search"
    description: str = (
        "Search the property database using natural language. "
        "Returns the most relevant property listings for a given query. "
        "Use this to find properties matching criteria like location, price, size, or features."
    )

    def _run(self, query: str) -> str:
        results = search(query, n_results=3)

        if not results:
            return "No matching properties found."

        lines = ["USE ONLY THESE PROPERTY IDs IN YOUR RESPONSE:\n"]
        for match in results:
            pid = match["property_id"]
            snippet = match["content"].replace("\n", " ").strip()[:300]
            lines.append(f"PROPERTY_ID={pid} | {snippet}")

        return "\n\n".join(lines)