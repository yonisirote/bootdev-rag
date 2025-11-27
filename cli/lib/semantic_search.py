import os
from os import path
import re
import numpy as np

from sentence_transformers import SentenceTransformer

from lib.search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or text.isspace():
            raise ValueError("Input text cannot be empty.")
        embedding = self.model.encode([text])
        return embedding[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        doc_strings = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            doc_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
        np.save(path.join(CACHE_DIR, "movie_embeddings.npy"), self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(path.join(CACHE_DIR, "movie_embeddings.npy")):
            self.embeddings = np.load(path.join(CACHE_DIR, "movie_embeddings.npy"))
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` first.")
        embedded_query = self.generate_embedding(query)
        res = []
        for doc, emb in zip(self.documents, self.embeddings):
            similarity_score = cosine_similarity(embedded_query, emb)
            res.append((similarity_score, doc))
        res.sort(key=lambda x: x[0], reverse=True)
        return [{"score": score, "title": doc["title"], "description": doc["description"]} for score, doc in res[:limit]]
    
        

def verify_model():
    semantic_search = SemanticSearch()
    model = semantic_search.model
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    
def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")   
    print(f"Shape: {embedding.shape}")
    
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def search_command(query, limit=DEFAULT_SEARCH_LIMIT):
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_embeddings(documents)
    results = semantic_search.search(query, limit)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['description']}")
    
def chunk_command(text: str, chunk_size, overlap=0):
    words = text.split()
    chunks = [] 
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")
        
def semantic_chunk_command(text: str, max_chunk_size, overlap=0):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    for i in range(0, len(sentences), max_chunk_size - overlap):
        chunk = ' '.join(sentences[i:i + max_chunk_size])
        chunks.append(chunk)
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")
    
    