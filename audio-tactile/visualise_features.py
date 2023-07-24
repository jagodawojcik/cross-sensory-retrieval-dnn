import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Load embeddings
tactile_embeddings = np.load("trained_tactile_embeddings.npy", allow_pickle=True).item()
audio_embeddings = np.load("trained_audio_embeddings.npy", allow_pickle=True).item()

def prepare_data(embeddings):
    data = []
    labels = []
    for label, embedding in embeddings.items():
        data.extend(embedding)
        labels.extend([label]*len(embedding))
    return np.array(data), np.array(labels)

def plot_embeddings():
    # Prepare the data
    tactile_data, tactile_labels = prepare_data(tactile_embeddings)
    audio_data, audio_labels = prepare_data(audio_embeddings)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    tactile_data_2d = tsne.fit_transform(tactile_data)
    audio_data_2d = tsne.fit_transform(audio_data)

    # Visualize the results
    plt.figure(figsize=(10, 10))

    # Get unique labels and assign each one a color
    unique_labels = list(set(tactile_labels)) # Assumes same labels exist in tactile data
    colors = plt.cm.get_cmap('rainbow', len(unique_labels))

    for i, label in enumerate(unique_labels):
        plt.scatter(tactile_data_2d[tactile_labels==label, 0], tactile_data_2d[tactile_labels==label, 1], color=colors(i), label=f"Tactile-{label}", alpha=0.6)
        plt.scatter(audio_data_2d[audio_labels==label, 0], audio_data_2d[audio_labels==label, 1], color=colors(i), label=f"Audio-{label}", alpha=0.6, marker='x')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_embeddings()
