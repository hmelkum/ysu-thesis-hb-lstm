import torch


def create_lagged_news_sequences(embeddings, window=5):
    n_days = embeddings.shape[0]
    sequences = []

    for i in range(window, n_days):
        seq = embeddings[i - window + 1 : i + 1]
        sequences.append(seq)

    return torch.stack(sequences)
