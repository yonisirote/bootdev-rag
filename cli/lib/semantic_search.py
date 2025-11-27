import os
from os import path
import numpy as np

from sentence_transformers import SentenceTransformer

from lib.search_utils import CACHE_DIR, load_movies


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