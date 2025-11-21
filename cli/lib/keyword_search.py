import string

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords
from nltk.stem import PorterStemmer


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    cleaned_query = clean_string(query)
    query_tokens = tokenize(cleaned_query)
    for movie in movies:
        cleaned_title = clean_string(movie["title"])
        title_tokens = tokenize(cleaned_title)
        for qtoken in query_tokens:
            for ttoken in title_tokens:
                if qtoken in ttoken:
                    results.append(movie)
                    if len(results) >= limit:
                        break   
    return results


def clean_string(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def tokenize(text: str) -> list[str]:
    tokens = text.split()
    stop_words = load_stopwords()
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens if token and token not in stop_words]