import argparse

from lib.augmented_generation import rag


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize text using RAG"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Generate answer with citations using RAG"
    )
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    question_parser = subparsers.add_parser(
        "question", help="Answer a question using RAG"
    )
    question_parser.add_argument("question", type=str, help="Question for RAG")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            result = rag(query, args.command, args.limit)
            if result is None:
                print("Error: No results returned from RAG.")
            else:
                print("Search Results:")
                for res in result["docs"]:
                    print(f"    - {res['title']}")
                print("\n")
                print("RAG RESPONSE:")
                print(result["response"].text or "No response generated.")
        case "summarize":
            query = args.query
            result = rag(query, args.command, args.limit)
            if result is None:
                print("Error: No results returned from RAG.")
            else:
                print("Search Results:")
                for res in result["docs"]:
                    print(f"    - {res['title']}")
                print("\n")
                print("LLM Summary:")
                print(result["response"].text or "No response generated.")
        case "citations":
            query = args.query
            result = rag(query, args.command, args.limit)
            if result is None:
                print("Error: No results returned from RAG.")
            else:
                print("Search Results:")
                for res in result["docs"]:
                    print(f"    - {res['title']}")
                print("\n")
                print("LLM Answer:")
                print(result["response"].text or "No response generated.")
        case "question":
            question = args.question
            result = rag(question, args.command, args.limit)
            if result is None:
                print("Error: No results returned from RAG.")
            else:
                print("Search Results:")
                for res in result["docs"]:
                    print(f"    - {res['title']}")
                print("\n")
                print("Answer:")
                print(result["response"].text or "No response generated.")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
