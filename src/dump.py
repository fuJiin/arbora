# Dump from Gemini
#
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt

class SDREncoder:
    def __init__(self, model_name="gpt2", n=2048, k=40):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.n = n
        self.k = k
        self.token_to_sdr = {}

    def get_sdr(self, token_id):
        if token_id not in self.token_to_sdr:
            rng = np.random.default_rng(token_id)
            self.token_to_sdr[token_id] = set(rng.choice(self.n, self.k, replace=False))
        return self.token_to_sdr[token_id]

    def encode_text(self, text):
        return self.tokenizer.encode(text)

class SDRModel:
    def __init__(self, n=2048, k=40, max_lr=0.5, weight_decay=0.999):
        self.n = n
        self.k = k
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.weights = {} # Sparse Map: {source_bit: array_of_weights}
        self.history = {} # {t: sdr_indices}

    def predict(self, t):
        prediction_vector = np.zeros(self.n)
        for i in range(max(0, t-101), t):
            past_sdr = self.history[i]
            strength = 1 - ((t - i) / 101)
            for bit_idx in past_sdr:
                if bit_idx in self.weights:
                    prediction_vector += self.weights[bit_idx] * strength
        
        # Local Normalization
        max_val = np.max(prediction_vector)
        if max_val > 0:
            prediction_vector /= max_val
            
        top_k_indices = np.argpartition(prediction_vector, -self.k)[-self.k:]
        return set(top_k_indices)

    def update(self, t, current_sdr, predicted_sdr):
        overlap = len(current_sdr.intersection(predicted_sdr))
        iou = overlap / self.k
        actual_eta = self.max_lr * (1.0 - iou)
        
        for i in range(max(0, t-101), t):
            past_indices = self.history[i]
            trace_strength = 1 - ((t - i) / 101)
            
            for p_idx in past_indices:
                if p_idx not in self.weights:
                    self.weights[p_idx] = np.zeros(self.n)
                
                # Apply Weight Decay
                self.weights[p_idx] *= self.weight_decay
                
                # Reinforce & Penalize
                for c_idx in current_sdr:
                    self.weights[p_idx][c_idx] += actual_eta * trace_strength
                for f_idx in (predicted_sdr - current_sdr):
                    self.weights[p_idx][f_idx] -= actual_eta * trace_strength * 0.5
        return iou

# Main Execution Logic
encoder = SDREncoder()
model = SDRModel()
dataset = load_dataset("roneneldan/TinyStories", streaming=True, split="train")

t = 0
accuracies = []

print("Starting training on TinyStories...")
for story in dataset:
    token_ids = encoder.encode_text(story["text"])
    for tid in token_ids:
        current_sdr = encoder.get_sdr(tid)
        
        if t > 0:
            pred = model.predict(t)
            iou = model.update(t, current_sdr, pred)
            accuracies.append(iou)
            if t % 100 == 0:
                print(f"Token {t} | Rolling IoU: {np.mean(accuracies[-100:]):.4f}")
        
        model.history[t] = current_sdr
        if t > 101: del model.history[t-101]
        t += 1
    if t > 5000: break # Small limit for testing

