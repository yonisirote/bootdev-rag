from sentence_transformers import SentenceTransformer


class SemanticSearch:

    def __init__(self):
        self.model = model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text):
        if not text:
            raise ValueError("Input text cannot be empty.")
        embedding = self.model.encode([text])
        return embedding[0]



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
    
    

    
    