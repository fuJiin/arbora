import numpy as np

# TODO: update functions to accept these params as local vars

N = 2048 # number of SDR columns
X = 101 # eligibility window in time steps
P = 0.5 # false positive penalty

def compute_iou(current_sdr, pred_sdr):
    overlap = len(current_sdr.intersection(pred_sdr))
    return overlap / len(current_sdr)


def compute_trace_strength(t, i, x):
    """
    Linear decay based on how long ago the trace was
    t: current time
    i: historical time
    x: total eligbile time
    """
    assert (t - 1) <= x
    return 1 - ((t - i)/x)


def get_normalized_top_k(pred_vector, k):
    norm_vector = local_normalize(pred_vector)
    top_k_indices = np.argpartition(norm_vector, -k)[-k:]
    return set(top_k_indices), pred_vector



def update_weights(weights, history, t, current_sdr, pred_sdr, max_lr, iou):
    # Adaptive Learning Rate: eta = max_lr * surprise
    # TODO: surprise can be replaced with some other global signal (reward)
    actual_eta = max_lr * (1.0 - iou) # 
    
    # Competitive Update: look back through the X-step window
    for i in range(max(0, t-X), t):
        past_indices = history[i]
        # Linear Decay Trace
        strength = compute_trace_strength(t, i, X)
        
        for p_idx in past_indices:
            # Lazy Init: Ensure the row exists in our sparse weight dict
            if p_idx not in weights:
                weights[p_idx] = np.zeros(N)
            
            # Reinforce bits that SHOULD have been active
            # TODO: we should probably have a ceiling
            for c_idx in current_sdr:
                weights[p_idx][c_idx] += actual_eta * strength
            
            # Penalize bits that were WRONGLY active (False Positives)
            # TODO: add floor by parameter
            false_positives = pred_sdr - current_sdr
            for f_idx in false_positives:
                weights[p_idx][f_idx] -= actual_eta * strength * P # Gentler penalty
                
    return iou



def local_normalize(vector):
    max_val = np.max(vector)
    return vector if max_val > 0 else vector / max_val 
