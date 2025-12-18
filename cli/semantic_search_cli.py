#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    chunk_command,
    embed_chunks_command,
    embed_query_text,
    embed_text,
    search_chunked_command,
    search_command,
    semantic_chunk_command,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the model")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate embedding for text"
    )
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    embedquery_parser = subparsers.add_parser(
        "embedquery", help="Generate embedding for query text"
    )
    embedquery_parser.add_argument("query", type=str, help="Query text to embed")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Limit the number of results"
    )

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--overlap", type=int, help="Size of each chunk overlap")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Size of each chunk"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk text semantically"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="Size of each chunk"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Size of each chunk overlap"
    )

    subparsers.add_parser("embed_chunks", help="Embed semantic chunks")

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search using chunked semantic search"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="Limit the number of results"
    )

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
            search_command(args.query, args.limit)
        case "chunk":
            print(f"Chunking {len(args.text)} characters")
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            print(f"Semantically chunking {len(args.text)} characters")
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            print("Embedding semantic chunks...")
            embed_chunks_command()
        case "search_chunked":
            print(f"Searching chunked for: {args.query} (limit: {args.limit})")
            search_chunked_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
