import numpy as np

# 1. Setup: 3 tokens, each with a 4-dimension embedding (d=4)
# Let's imagine: [ "The", "robot", "it" ]
embeddings = np.array([
    [1, 0, 0, 0], # "The"
    [0, 1, 0, 0], # "robot"
    [0, 0.9, 0, 0.1] # "it" (already slightly similar to robot)
])

# 2. Linear Transformations (Wq, Wk, Wv)
# For this demo, let's assume they are identity matrices (no change)
Q = embeddings  # Queries
K = embeddings  # Keys
V = embeddings  # Values

# 3. Calculate Raw Scores (Q * K-transpose)
# This measures similarity between every word and every other word
scores = np.matmul(Q, K.T)
output = np.matmul(scores, V)

print("Raw Similarity Scores:\n", scores)
print("Output:\n", output)
