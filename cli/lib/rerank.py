import json
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
        res["individual_score"] = score
    reranked = sorted(results, key=lambda x: x.get("individual_score", 0), reverse=True)
    return reranked


def rerank_batch(query: str, results: list[dict]) -> list[dict]:
    doc_map = {res["id"]: res for res in results}
    doc_list_str = "\n\n".join(
        [
            f"{res.get('id', '')}: {res.get('title', '')} - {res.get('document', '')}"
            for res in results
        ]
    )
    prompt = f"""Rank these movies by relevance to the search query.

    Query: "{query}"

    Movies:
    {doc_list_str}

    Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

    [75, 12, 34, 2, 1]

    Do not include any text other than the JSON list. Do not include the word "json", or any quotes.
    """

    response = client.models.generate_content(model=model, contents=prompt)
    ranked_ids_text = (response.text or "").strip()
    print(f"Rerank response: {ranked_ids_text}")
    try:
        ranked_ids = json.loads(ranked_ids_text)
        # ranked_ids = [str(i) for i in ranked_ids]  # Normalize to strings
    except (json.JSONDecodeError, ValueError):
        print("Failed to parse rerank response as JSON.")
        ranked_ids = []  # Default to empty if parsing fails
    for rank, doc_id in enumerate(ranked_ids, 1):
        if doc_id in doc_map:
            doc_map[doc_id]["batch_rank"] = rank
            print(f"Doc ID {doc_id} assigned rerank score {rank}")
    reranked = sorted(results, key=lambda x: x.get("batch_rank", float("inf")))
    return reranked


def rerank(query, results, rerank_method):
    match rerank_method:
        case "individual":
            return rerank_individual(query, results)
        case "batch":
            return rerank_batch(query, results)
        case _:
            return results
