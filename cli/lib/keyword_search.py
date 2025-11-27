import math
import os
import pickle
import string
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords

BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        cleaned_text = clean_string(text)
        tokens = tokenize(cleaned_text)
        self.doc_lengths[doc_id] = len(tokens)
        self.term_frequencies[doc_id] = Counter(tokens)
        for token in tokens:
            self.index[token].add(doc_id)
    
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths)

    def get_documents(self, term):
        token = tokenize(term)
        if len(token) > 1:
            raise ValueError("Term must be a single word")
        if token[0] in self.index:
            return sorted(list(self.index[token[0]]))
        return []
    
    def get_tf(self, doc_id, term):
        token = tokenize(term)
        if len(token) > 1:
            raise ValueError("Term must be a single word")
        return self.term_frequencies[doc_id][token[0]]
    
    def get_bm25_idf(self, term: str) -> float:
        token = tokenize(term)
        if len(token) > 1:
            raise ValueError("Term must be a single word")
        doc_count = len(self.docmap)
        doc_freq = len(self.index[token[0]])
        return math.log((doc_count - doc_freq + 0.5) / (doc_freq + 0.5)+ 1)
        
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)
        length_norm = (1 - b + b * doc_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def bm25(self, doc_id, term) -> float:
        bm25_idf = self.get_bm25_idf(term)
        bm25_tf = self.get_bm25_tf(doc_id, term)
        return bm25_idf * bm25_tf
        
    def bm25_search(self, query, limit):
        tokens = tokenize(query)
        scores = {}
        for doc in self.docmap.keys():
            score = 0.0
            for token in tokens:
                score += self.bm25(doc, token)
            if score > 0:
                scores[doc] = score
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs[:limit]
        

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
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        index_path = os.path.join(CACHE_DIR, "index.pkl")
        docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Cache file not found: {index_path}")
        if not os.path.exists(docmap_path):
            raise FileNotFoundError(f"Cache file not found: {docmap_path}")

        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)


def build_command():
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()

def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def idf_command(term):
    idx = InvertedIndex()
    idx.load()
    doc_count = len(idx.docmap)
    term_doc_count = len(idx.get_documents(term))
    return math.log((doc_count + 1) / (term_doc_count + 1))
    
def tfidf_command(doc_id, term):
    rf = tf_command(doc_id, term)
    idf = idf_command(term)
    return rf * idf

def bm25_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    idx = InvertedIndex()
    idx.load()
    res = idx.bm25_search(query, limit)
    return [(res[0], idx.docmap[res[0]]["title"], res[1]) for res in res]

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    results = []
    query_tokens = tokenize(query)
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
        stemmer.stem(clean_string(token)) for token in tokens if token and token not in stop_words
    ]
