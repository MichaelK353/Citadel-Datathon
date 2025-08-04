import os
from types import SimpleNamespace

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sbert = SimpleNamespace(
    # Parameters for encoder
    encode_batch_size=256,
    # Parameters for paraphrase mining
    paraphrase_query_chunk_size=100,
    paraphrase_corpus_chunk_size=500,
    paraphrase_batch_size=32,
    # Parameters for semantic search
    semantic_corpus_chunk_size=500,
)
