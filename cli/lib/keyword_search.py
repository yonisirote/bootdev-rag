import os
import pickle
import string
from collections import defaultdict

from nltk.stem import PorterStemmer

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "cache")


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}

    def __add_document(self, doc_id, text):
        cleaned_text = clean_string(text)
        tokens = tokenize(cleaned_text)
        for token in tokens:
            self.index[token].add(doc_id)

    def get_documents(self, term):
        term = term.lower()
        if term in self.index:
            return sorted(list(self.index[term]))
        return []

    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        with open(os.path.join(CACHE_DIR, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)

    def load(self):
        index_path = os.path.join(CACHE_DIR, "index.pkl")
        docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Cache file not found: {index_path}")
        if not os.path.exists(docmap_path):
            raise FileNotFoundError(f"Cache file not found: {docmap_path}")

        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)


def build_command():
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    results = []
    cleaned_query = clean_string(query)
    query_tokens = tokenize(cleaned_query)
    for token in query_tokens:
            for doc_id in idx.get_documents(token):
                results.append(idx.docmap[doc_id])
                if len(results) >= limit:
                        return results
    return results


def clean_string(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def tokenize(text: str) -> list[str]:
    tokens = text.split()
    stop_words = load_stopwords()
    stemmer = PorterStemmer()
    return [
        stemmer.stem(token) for token in tokens if token and token not in stop_words
    ]
