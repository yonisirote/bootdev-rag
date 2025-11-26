#!/usr/bin/env python3

import argparse

from lib.semantic_search import embed_text, verify_model

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("verify", help="Verify the model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    args = parser.parse_args()

    match args.command:
        case "verify":
            print("Verifying the semantic search model...")
            verify_model()
        case "embed_text":
            print("Generating embedding for text...")
            embed_text(args.text)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()