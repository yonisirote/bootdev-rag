import os

from dotenv import load_dotenv
from google import genai

from .hybrid_search import HybridSearch
from .search_utils import load_movies

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def get_results(query, limit):
    movies = load_movies()
    searcher = HybridSearch(movies)
    results = searcher.rrf_search(query, k=60, limit=5)
    return results


def rag_command(query, limit=5):
    results = get_results(query, limit)

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Documents:
    {results}

    Provide a comprehensive answer that addresses the query:"""

    response = client.models.generate_content(model=model, contents=prompt)

    return {"docs": results, "response": response}


def sumarize_command(query, limit=5):
    results = get_results(query, limit)

    prompt = f"""
    Provide information useful to this query by synthesizing information from multiple search results in detail.
    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
    This should be tailored to Hoopla users. Hoopla is a movie streaming service.
    Query: {query}
    Search Results:
    {results}
    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
    """

    response = client.models.generate_content(model=model, contents=prompt)
    return {"docs": results, "response": response}


def citations_command(query, limit):
    results = get_results(query, limit)

    prompt = f"""Answer the question or provide information based on the provided documents.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

    Query: {query}

    Documents:
    {results}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources using [1], [2], etc. format when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)

    return {"docs": results, "response": response}


def question_command(query, limit):
    results = get_results(query, limit)

    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Question: {query}

    Documents:
    {results}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation

    Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)

    return {"docs": results, "response": response}


def rag(query, command, limit=5):
    match command:
        case "rag":
            return rag_command(query, limit)
        case "summarize":
            return sumarize_command(query, limit)
        case "citations":
            return citations_command(query, limit)
        case "question":
            return question_command(query, limit)
