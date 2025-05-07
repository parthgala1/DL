import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re

from TCNN import TCNN

# --- Data Preparation ---
class SentenceDataset(Dataset):
    def __init__(self, csv_path, max_len=320, t_dim=5):
        df = pd.read_csv(csv_path)
        self.sentences = df['sentence'].apply(lambda x: re.sub(r"b[\"']", "", str(x)).strip()).tolist()
        self.max_len = max_len
        self.t_dim = t_dim

        # Simple char-level encoding for demonstration
        self.vocab = {c: i+1 for i, c in enumerate(sorted(set(''.join(self.sentences))))}
        self.vocab['<PAD>'] = 0

        self.encoded = [self.encode(s) for s in self.sentences]

    def encode(self, sentence):
        ids = [self.vocab.get(c, 0) for c in sentence]
        if len(ids) < self.max_len * self.t_dim:
            ids += [0] * (self.max_len * self.t_dim - len(ids))
        else:
            ids = ids[:self.max_len * self.t_dim]
        return ids

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        arr = np.array(self.encoded[idx], dtype=np.float32)
        arr = arr.reshape(1, self.t_dim, self.max_len)  # (1, T, 320)
        return torch.tensor(arr)

# --- Feature Extraction Helper ---
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # Forward pass up to the last Conv2d_7 layer
            x = model.Conv2d_1(batch)
            x = model.Conv2d_2(x)
            x = model.Conv2d_3(x)
            x = model.Conv2d_4(x)
            x = model.Conv2d_5(x)
            x = model.Conv2d_6(x)
            x = model.Conv2d_7(x)
            # Global average pooling over spatial dims
            pooled = x.mean(dim=[2, 3]).cpu().numpy()
            features.append(pooled)
    return np.vstack(features)

# --- Main Pipeline ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64

    # Load datasets
    train_dataset = SentenceDataset('train.csv')
    test_dataset = SentenceDataset('test.csv', max_len=train_dataset.max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load TCNN model
    model = TCNN().to(device)

    # (Optional) Train the model here if you have labels and a loss function
    # For clustering, we just extract features from the pretrained/untrained model

    # Extract features
    train_features = extract_features(model, train_loader, device)
    test_features = extract_features(model, test_loader, device)

    # --- Clustering ---
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    train_clusters = kmeans.fit_predict(train_features)
    test_clusters = kmeans.predict(test_features)

    # --- Visualization ---
    tsne = TSNE(n_components=2, random_state=42)
    train_embedded = tsne.fit_transform(train_features)
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        idx = train_clusters == i
        plt.scatter(train_embedded[idx, 0], train_embedded[idx, 1], label=f'Cluster {i}', alpha=0.5)
    plt.title('TCNN Sentence Embedding Clusters (Train)')
    plt.legend()
    plt.savefig('train_clusters.png')
    plt.show()

    # Plot test clusters
    test_embedded = tsne.fit_transform(test_features)
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        idx = test_clusters == i
        plt.scatter(test_embedded[idx, 0], test_embedded[idx, 1], label=f'Cluster {i}', alpha=0.5)
    plt.title('TCNN Sentence Embedding Clusters (Test)')
    plt.legend()
    plt.savefig('test_clusters.png')
    plt.show()

    # Save cluster assignments
    pd.read_csv('train.csv').assign(cluster=train_clusters).to_csv('train_with_clusters.csv', index=False)
    pd.read_csv('test.csv').assign(cluster=test_clusters).to_csv('test_with_clusters.csv', index=False)

if __name__ == '__main__':
    main()