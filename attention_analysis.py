import numpy as np
import matplotlib.pyplot as plt

def visualize_attention():
    weights = np.load("attention_weights.npy")
    avg_weights = weights.mean(axis=0)

    plt.figure(figsize=(8,4))
    plt.plot(avg_weights)
    plt.title("Average Attention Weights Across Time Steps")
    plt.xlabel("Time Step")
    plt.ylabel("Attention Weight")
    plt.tight_layout()
    plt.savefig("attention_weights.png")
