import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Load embeddings
visual_embeddings = np.load("visual_embeddings_kaggle_train.npy", allow_pickle=True).item()
tactile_embeddings = np.load("tactile_embeddings_kaggle_train.npy", allow_pickle=True).item()

def prepare_data(embeddings):
    data = []
    labels = []
    for label, embedding in embeddings.items():
        data.extend(embedding)
        labels.extend([label]*len(embedding))
    return np.array(data), np.array(labels)

   
def visualize_embeddings(visual_embeddings, tactile_embeddings):
    # Prepare the data
    visual_data, visual_labels = prepare_data(visual_embeddings)
    tactile_data, tactile_labels = prepare_data(tactile_embeddings)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    visual_data_2d = tsne.fit_transform(visual_data)
    tactile_data_2d = tsne.fit_transform(tactile_data)

    # Visualize the results
    plt.figure(figsize=(10, 10))

    # Get unique labels and assign each one a color
    unique_labels = list(set(visual_labels)) # Assumes same labels exist in tactile data
    colors = plt.cm.get_cmap('rainbow', len(unique_labels))

    for i, label in enumerate(unique_labels):
        plt.scatter(visual_data_2d[visual_labels==label, 0], visual_data_2d[visual_labels==label, 1], color=colors(i), label=f"Visual-{label}", alpha=0.6)
        plt.scatter(tactile_data_2d[tactile_labels==label, 0], tactile_data_2d[tactile_labels==label, 1], color=colors(i), label=f"Tactile-{label}", alpha=0.6, marker='x')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    visualize_embeddings(visual_embeddings, tactile_embeddings)
