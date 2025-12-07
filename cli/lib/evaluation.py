import json

from .hybrid_search import HybridSearch
from .search_utils import GOLDEN_SET_PATH, load_movies
from .semantic_search import SemanticSearch


def check_precision(limit):
    with open(GOLDEN_SET_PATH, "r") as f:
        golden_dataset = json.load(f)

    test_cases = golden_dataset["test_cases"]

    movies = load_movies()
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)

    searcher = HybridSearch(movies)

    print(f"{limit}\n")

    for entry in test_cases:
        query = entry["query"]
        expected_results = entry["relevant_docs"]

        actual_results = searcher.rrf_search(query, k=60, limit=limit)
        actual_titles = {res["title"] for res in actual_results}

        correct_count = set(expected_results).intersection(actual_titles)
        precision = len(correct_count) / limit

        print(f"- Query: {query}")
        print(f"    - Precision@{limit}: {precision:.4f}")
        print(f"    - Retrieved: {', '.join(actual_titles)}")
        print(f"    - Relevant: {', '.join(expected_results)}")
