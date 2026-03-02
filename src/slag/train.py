import numpy as np
from slag.core import N, X,compute_trace_strength, get_normalized_top_k, compute_iou, update_weights


K = 40 # top K bits we strengthen
# L =   # global surprise threshold, which triggers learning


weights = np.random.randn(N, N) * 0.01
history = {} # {t, SDR}
rolling_accuracy = []

for t, token in tiny_stories_stream:
    # 1. Encode word to SDR
    current_sdr_indices = encoder.get_active_indices(token)

    # 2: Prediction step (skip if t == 0)
    if t > 0:
        pred_vector = np.zeros(N)
        
        # Look back through window
        for i in range(max(0, t-X), t):
            past_sdr = history[i]

            # Apply our Linear Decay trace
            trace_strength = compute_trace_strength(t, i, X)

            for bit_idx in past_sdr:
                pred_vector += weights[bit_idx] * trace_strength

        # Local normalization & Top-K
        pred_sdr = get_normalized_top_k(pred_vector, K)

        # 3. Evaluation & weight updates
        # TODO: actually 
        rolling_accuracy.append(compute_iou(current_sdr, pred_sdr))
        update_weights()

    # 4. Update history
    history[t] = current_sdr_indices

    if t > X:
        del history[t-X]
