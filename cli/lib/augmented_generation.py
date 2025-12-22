import os

from dotenv import load_dotenv
from google import genai

from .hybrid_search import rrf_search_command

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def rag_command(query):
    search_results = rrf_search_command(query, limit=5)

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Documents:
    {search_results["results"]}

    Provide a comprehensive answer that addresses the query:"""

    response = client.models.generate_content(model=model, contents=prompt)

    print("Search Results:")
    for res in search_results["results"]:
        print(f"    - {res['title']}")
    print("\n")
    print("RAG RESPONSE:")
    print(response.text or "No response generated.")
