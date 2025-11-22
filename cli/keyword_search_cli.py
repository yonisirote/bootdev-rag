#!/usr/bin/env python3

import argparse

from lib.inverted_index import InvertedIndex
from lib.keyword_search import search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build and save the inverted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case "build":
            print("Building inverted index...")
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()
            print("Index built and saved successfully!")

            # Get the first document ID for the token 'merida'
            merida_docs = inverted_index.index.get("merida")
            if merida_docs:
                first_doc_id = min(merida_docs)
                print(f"First document ID for token 'merida': {first_doc_id}")
            else:
                print("Token 'merida' not found in index")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
