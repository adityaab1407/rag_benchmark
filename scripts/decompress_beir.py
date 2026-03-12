import gzip
import shutil

pairs = [
    ("data/raw/beir/corpus.jsonl.gz", "data/raw/beir/scifact_corpus.jsonl"),
    ("data/raw/beir/queries.jsonl.gz", "data/raw/beir/scifact_queries.jsonl"),
]
for src, dst in pairs:
    print(f'Decompressing {src} -> {dst}')
    with gzip.open(src, 'rb') as f_in, open(dst, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print('Done')
