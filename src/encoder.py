from transformers import AutoTokenizer
import numpy as np

class SDREncoder:
    def __init__(self, model_name="gpt2", n=2048, k=40):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.n = n
        self.k = k
        # Map each token ID to a fixed set of random indices
        self.token_to_sdr = {}

    def get_sdr(self, token_id):
        if token_id not in self.token_to_sdr:
            # Seed with token_id for consistency
            rng = np.random.default_rng(token_id)
            self.token_to_sdr[token_id] = set(rng.choice(self.n, self.k, replace=False))
        return self.token_to_sdr[token_id]

    def encode(self, text):
        ids = self.tokenizer.encode(text)
        return [self.get_sdr(tid) for tid in ids]
