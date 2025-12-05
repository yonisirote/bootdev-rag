import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def rerank_individual(query: str, results: list[dict]) -> list[dict]:
    for res in results:
        time.sleep(3)  # To avoid rate limiting
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {res.get("title", "")} - {res.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Give me ONLY the number in your response, no other text or explanation.

        Score:"""

        response = client.models.generate_content(model=model, contents=prompt)
        score_text = (response.text or "").strip()
        try:
            score = float(score_text)
        except ValueError:
            score = 0.0  # Default to 0 if parsing fails
        res["rerank_score"] = score
    reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
    return reranked


def rerank(query, results, rerank_method):
    if rerank_method == "individual":
        return rerank_individual(query, results)
    else:
        raise ValueError(f"Unknown rerank method: {rerank_method}")
