#!/usr/bin/env python3
import argparse

from lib.keyword_search import  search_command, build_command, tf_command, idf_command, tfidf_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build and save the inverted index")
    
    tf_parser = subparsers.add_parser("tf", help="Get term frequency in document")
    tf_parser.add_argument("document_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to search for")
    
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term to search for")
    
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score")
    tfidf_parser.add_argument("document_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to search for")
    
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case "build":
            print("Building inverted index...")
            build_command()
            print("Index built and saved successfully!")
        case "tf":
            print("Getting term frequency...")
            tf = tf_command(args.document_id, args.term)
            print(f"Term frequency: {tf}")
        case "idf":
            print("Getting inverse document frequency...")
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            print("Getting TF-IDF score...")
            tfidf = tfidf_command(args.document_id, args.term)
            print(f"TF-IDF score: {tfidf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
