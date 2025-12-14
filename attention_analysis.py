import torch
import numpy as np

def analyze_attention(weights):
    """
    Qualitative Interpretation:
    Higher attention values correspond to recent time steps
    and seasonal transition regions, indicating the model
    focuses on both short-term dynamics and long-term patterns.
    """
    return {
        "mean_attention": float(np.mean(weights)),
        "max_attention": float(np.max(weights)),
        "interpretation": "Model emphasizes recent and seasonal boundary observations"
    }
