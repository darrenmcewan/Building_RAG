#!/usr/bin/env python3

import argparse
from unittest import case
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify", help="Verify loading of the semantic search model"
    )

    embed_parser = subparsers.add_parser(
        "embed_text", help="Generate embeddings for the input text"
    )
    embed_parser.add_argument("text", type=str, help="Text to embed")

    verify_embed_parser = subparsers.add_parser(
        "verify_embeddings", help="Verify loading or creation of embeddings for documents"
    )
    verify_embed_parser.set_defaults(func=verify_embeddings)

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate embeddings for a query text"
    )
    embed_query_parser.add_argument("text", type=str, help="Query text to embed")

    search_parser = subparsers.add_parser(
        "search", help="Search for similar documents based on a query"
    )
    search_parser.add_argument("text", type=str, help="Query text to search for")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of top similar documents to return"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.text)
        case "search":
                from lib.semantic_search import SemanticSearch, load_movies
                search_instance = SemanticSearch()
                documents = load_movies()
                search_instance.load_or_create_embeddings(documents)
                results = search_instance.search(args.text, args.limit)
                for i, (similarity, title, description) in enumerate(results):
                    print(f"{i+1}: {title} (score: {similarity:.4f})")
                    print(f"{description}")
                    print("")
  
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()