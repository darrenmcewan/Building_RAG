# RAG Search Engine

A dual-mode search engine demonstrating both keyword-based (BM25) and semantic search approaches for querying movie data.

## Overview

This project implements two independent search systems operating on a dataset of 25,000+ movies:
- **Keyword Search**: Traditional BM25 ranking with inverted index
- **Semantic Search**: Neural embeddings with cosine similarity using sentence-transformers

## Setup

### Prerequisites
- Python 3.14 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG
```

2. Install dependencies:
```bash
pip install -e .
```

This will install:
- `sentence-transformers`: For semantic embeddings
- `nltk`: For text tokenization and stemming
- `numpy`: For vector operations

### First-Time Initialization

**Keyword Search** requires building the inverted index before use:
```bash
python -m cli.keyword_search_cli build
```

**Semantic Search** automatically generates embeddings on first use (cached for subsequent searches).

## Usage

### Keyword Search CLI

The keyword search CLI provides BM25-based search and analysis tools.

#### Basic Search
```bash
# Search using BM25 algorithm
python -m cli.keyword_search_cli bm25search "action thriller"

# Simple search (returns first matches containing query terms)
python -m cli.keyword_search_cli search "space adventure"
```

#### Analysis Commands

Get term frequency for a document:
```bash
python -m cli.keyword_search_cli tf 42 "adventure"
```

Get inverse document frequency for a term:
```bash
python -m cli.keyword_search_cli idf "space"
```

Get TF-IDF score:
```bash
python -m cli.keyword_search_cli tfidf 42 "adventure"
```

Get BM25 IDF score:
```bash
python -m cli.keyword_search_cli bm25idf "thriller"
```

Get BM25 TF score with custom parameters:
```bash
python -m cli.keyword_search_cli bm25tf 42 "adventure" 1.5 0.75
```

### Semantic Search CLI

The semantic search CLI uses neural embeddings to find conceptually similar movies.

#### Search for Movies
```bash
# Search with default limit (5 results)
python -m cli.semantic_search_cli search "movies about time travel"

# Search with custom limit
python -m cli.semantic_search_cli search "dystopian future" --limit 10
```

#### Verification Commands

Verify the model loads correctly:
```bash
python -m cli.semantic_search_cli verify
```

Verify embeddings are generated/loaded:
```bash
python -m cli.semantic_search_cli verify_embeddings
```

#### Embedding Analysis

Generate embedding for text:
```bash
python -m cli.semantic_search_cli embed_text "science fiction adventure"
```

Generate embedding for a query:
```bash
python -m cli.semantic_search_cli embedquery "space exploration"
```

## Key Differences: Keyword vs Semantic Search

### Keyword Search (BM25)
- **Requires exact term overlap** between query and documents
- **Fast** once index is built
- **Transparent scoring** - you can inspect TF, IDF, and BM25 components
- **Good for**: Specific title searches, known keywords

Example:
```bash
python -m cli.keyword_search_cli bm25search "alien invasion"
# Finds movies with words "alien" OR "invasion"
```

### Semantic Search
- **Finds conceptual matches** even without shared words
- **Slower** but more intelligent matching
- **Better for natural language** queries
- **Good for**: Descriptive searches, thematic queries

Example:
```bash
python -m cli.semantic_search_cli search "movies about extraterrestrial visitors"
# Finds movies about aliens, even if they don't use the word "extraterrestrial"
```

## Data

The movie dataset is located at `data/movies.json` and contains:
- **25,000+ movies** with title and description fields
- Each movie has a unique ID, title, and detailed description

## Cache System

Both search systems cache processed data in the `cache/` directory:

**Keyword Search Cache:**
- `index.pkl`: Inverted index mapping terms to documents
- `docmap.pkl`: Document metadata
- `term_frequencies.pkl`: Term frequency counts per document
- `doc_lengths.pkl`: Length of each document in tokens

**Semantic Search Cache:**
- `movie_embeddings.npy`: Pre-computed 384-dimensional embeddings for all movies

**Note**: If you modify `data/movies.json`, delete the `cache/` directory to regenerate.

## Project Structure

```
RAG/
├── cli/
│   ├── keyword_search_cli.py      # Keyword search CLI
│   ├── semantic_search_cli.py     # Semantic search CLI
│   └── lib/
│       ├── keyword_search.py      # BM25 implementation
│       ├── semantic_search.py     # Semantic search implementation
│       └── search_utils.py        # Shared utilities and constants
├── data/
│   ├── movies.json                # Movie dataset
│   └── stopwords.txt              # Stopwords for keyword search
├── cache/                         # Generated cache files
└── pyproject.toml                 # Project dependencies
```

## Troubleshooting

**"No module named 'cli'"**: Always run commands as modules using `python -m cli.X`, not as scripts.

**"No embeddings loaded"**: Call `verify_embeddings` first or ensure the search command loads embeddings via `load_or_create_embeddings()`.

**Stale results after data changes**: Delete the `cache/` directory and rebuild/regenerate.

**"Input text cannot be empty"**: Semantic search requires non-empty query strings.


