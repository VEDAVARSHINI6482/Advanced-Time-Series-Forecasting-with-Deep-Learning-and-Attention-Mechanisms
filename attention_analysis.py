import matplotlib.pyplot as plt
import numpy as np

def plot_attention(weights):
    avg_weights = np.mean(weights, axis=0)
    plt.figure(figsize=(8,4))
    plt.plot(avg_weights)
    plt.title("Average Attention Weights Across Time Steps")
    plt.xlabel("Time Step")
    plt.ylabel("Attention Weight")
    plt.tight_layout()
    plt.savefig("attention_weights.png")
