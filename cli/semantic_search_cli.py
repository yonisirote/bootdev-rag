#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_query_text, embed_text, search_command, verify_embeddings, verify_model

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("verify", help="Verify the model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")
    
    embedquery_parser = subparsers.add_parser("embedquery", help="Generate embedding for query text")
    embedquery_parser.add_argument("query", type=str, help="Query text to embed")
    
    search_parser = subparsers.add_parser("search", help="Search movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Limit the number of results")
    
    subparsers.add_parser("verify_embeddings", help="Verify embeddings generation")

    args = parser.parse_args()

    match args.command:
        case "verify":
            print("Verifying the semantic search model...")
            verify_model()
        case "embed_text":
            print("Generating embedding for text...")
            embed_text(args.text)
        case "verify_embeddings":
            print("Verifying embeddings...")
            verify_embeddings()
        case "embedquery":
            print("Generating embedding for query...")
            embed_query_text(args.query)
        case "search":
            print(f"Searching for: {args.query} (limit: {args.limit})")
            results = search_command(args.query, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"   {res['description']}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()