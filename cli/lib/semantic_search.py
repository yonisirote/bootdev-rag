from sentence_transformers import SentenceTransformer


class SemanticSearch:

    def __init__(self):
        self.model = self._initialize_model()


    def _initialize_model(self):
        # Load the model (downloads automatically the first time)
        model = SentenceTransformer('all-MiniLM-L6-v2')

        print(f"Model loaded: {model}")
        print(f"Max sequence length: {model.max_seq_length}")

        return model
    





def verify_model():
        semantic_search = SemanticSearch()
        model = semantic_search.model
        print(f"Model loaded: {model}")
        # Some models provide `max_seq_length`; guard access just in case.
        max_len = getattr(model, "max_seq_length", None)
        print(f"Max sequence length: {max_len}")


