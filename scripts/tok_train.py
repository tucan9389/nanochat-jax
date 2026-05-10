"""Train a tokenizer using our own BPE Tokenizer library, in the style of GPT-4."""

import os
import time
import argparse
import numpy as np

from nanochat_jax.tokenizer import RustBPETokenizer
from nanochat_jax.common import get_base_dir
from nanochat_jax.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 2B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode test"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# Cache a mapping from token id to number of bytes for that token. The bits-per-byte
# metric uses these counts to report a loss that is invariant to vocab size.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id]
    if token_str in special_set:
        token_bytes.append(0)  # special tokens are not counted
    else:
        id_bytes = len(token_str.encode("utf-8"))
        token_bytes.append(id_bytes)
token_bytes = np.array(token_bytes, dtype=np.int32)
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.npy")
np.save(token_bytes_path, token_bytes)
print(f"Saved token_bytes to {token_bytes_path}")

# -----------------------------------------------------------------------------
# Stats
token_bytes_nonzero = token_bytes[token_bytes > 0].astype(np.float32)
print("Tokenizer training stats:")
print(f"  num_special_tokens: {len(special_set)}")
print(f"  token_bytes_min: {int(token_bytes_nonzero.min())}")
print(f"  token_bytes_max: {int(token_bytes_nonzero.max())}")
print(f"  token_bytes_mean: {token_bytes_nonzero.mean():.4f}")
print(f"  token_bytes_std: {token_bytes_nonzero.std():.4f}")
