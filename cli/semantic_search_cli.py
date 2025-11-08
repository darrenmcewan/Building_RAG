#!/usr/bin/env python3

import argparse
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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()