from datasets import load_dataset

print('Starting BeIR scifact download...')

corpus = load_dataset("BeIR/scifact", "corpus")
corpus["corpus"].to_json("data/raw/beir/scifact_corpus.jsonl")

queries = load_dataset("BeIR/scifact", "queries")
queries["queries"].to_json("data/raw/beir/scifact_queries.jsonl")

print('Saved scifact corpus and queries to data/raw/beir')
