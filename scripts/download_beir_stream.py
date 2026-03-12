import json
from datasets import load_dataset

print('Starting streaming download of BeIR scifact...')

# Stream corpus
corpus_iter = load_dataset("BeIR/scifact", "corpus", split="train", streaming=True)
corpus_out = 'data/raw/beir/scifact_corpus.jsonl'
count = 0
with open(corpus_out, 'w', encoding='utf-8') as f:
    for item in corpus_iter:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        count += 1
        if count % 500 == 0:
            print(f'Wrote {count} corpus items')
print(f'Corpus streaming complete — total {count} items -> {corpus_out}')

# Stream queries
queries_iter = load_dataset("BeIR/scifact", "queries", split="train", streaming=True)
queries_out = 'data/raw/beir/scifact_queries.jsonl'
qcount = 0
with open(queries_out, 'w', encoding='utf-8') as f:
    for item in queries_iter:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        qcount += 1
        if qcount % 500 == 0:
            print(f'Wrote {qcount} queries')
print(f'Queries streaming complete — total {qcount} items -> {queries_out}')

print('BeIR streaming download finished')
