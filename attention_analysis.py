import torch

def extract_attention(model, sample):
    model.eval()
    with torch.no_grad():
        _, weights = model(sample.unsqueeze(0))
    return weights.squeeze().cpu().numpy()
