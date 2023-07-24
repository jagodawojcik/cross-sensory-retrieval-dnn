from collections import defaultdict
import random
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

EPOCHS = 400
BATCH_SIZE = 5
MARGIN = 1.2
HIDDEN_DIM = 250 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLPBranch(nn.Module):
    def __init__(self, input_dim=200, hidden_dim=HIDDEN_DIM, output_dim=50):
        super(MLPBranch, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.activation(x)
        return x


class TripletNetwork(nn.Module):
    def __init__(self, input_dim=200, hidden_dim=HIDDEN_DIM, output_dim=50):
        super(TripletNetwork, self).__init__()
        self.visual_branch = MLPBranch(input_dim, hidden_dim, output_dim)
        self.tactile_branch = MLPBranch(input_dim, hidden_dim, output_dim)
        
    def forward(self, visual_input, tactile_input, negative_input):
        visual_output = self.visual_branch(visual_input)
        tactile_output = self.tactile_branch(tactile_input)
        negative_output = self.tactile_branch(negative_input)
        
        return visual_output, tactile_output, negative_output

# Define triplet loss
class TripletLoss(nn.Module):
    def __init__(self, margin=MARGIN):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


def create_triplets(visual_embeddings, tactile_embeddings):
    triplets = []
    labels = list(visual_embeddings.keys())
    for label in labels:
        positives = list(zip(tactile_embeddings[label], visual_embeddings[label]))
        
        for i in range(len(positives)):
            anchor, positive = positives[i]
            negative_label = random.choice([l for l in labels if l != label])
            
            # Choose whether the negative example will be visual or tactile randomly
            if random.choice([True, False]):
                negative = visual_embeddings[negative_label][i % len(visual_embeddings[negative_label])]
            else:
                negative = tactile_embeddings[negative_label][i % len(tactile_embeddings[negative_label])]
                
            triplets.append((label, (anchor, positive, negative)))
            
    random.shuffle(triplets)
    return triplets


def train_with_triplet_loss(id, epochs=EPOCHS, batch_size=BATCH_SIZE):
    network = TripletNetwork()  
    network = network.to(device)

    # Get the triplet loss
    triplet_loss = TripletLoss()

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    # Load embeddings
    visual_embeddings = np.load("tactile_embeddings_kaggle.npy", allow_pickle=True).item()
    tactile_embeddings = np.load("visual_embeddings_kaggle.npy", allow_pickle=True).item()
    # Create triplets
    triplets = create_triplets(visual_embeddings, tactile_embeddings)

    # Convert to tensors
    triplets = [(torch.tensor(label), (torch.tensor(a), torch.tensor(p), torch.tensor(n))) for label, (a, p, n) in triplets]


    # Convert the list of triplets into a DataLoader
    from torch.utils.data import DataLoader, TensorDataset

    # Convert to TensorDataset
    triplets_dataset = TensorDataset(torch.tensor([label for label, _ in triplets]), 
                                    torch.stack([a for _, (a, _, _) in triplets]),
                                    torch.stack([p for _, (_, p, _) in triplets]),
                                    torch.stack([n for _, (_, _, n) in triplets]))


    triplets_loader = DataLoader(triplets_dataset, batch_size, shuffle=True)

    # Embeddings storage
    triplet_loss_visual_embeddings = defaultdict(list)
    triplet_loss_tactile_embeddings = defaultdict(list)
    # Training loop
    for epoch in range(EPOCHS):
        network.train()  # set network to training mode
        total_loss = 0
        for i, (label, anchor, positive, negative) in enumerate(triplets_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()

            # Get outputs and embeddings
            anchor_output, positive_output, negative_output = network(anchor, positive, negative)


            # Compute the loss
            loss = triplet_loss(anchor_output, positive_output, negative_output)
            total_loss += loss.item()

            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Save embeddings along with their labels only in the last epoch
            if epoch == epochs - 1:
                for j in range(anchor_output.shape[0]):
                    label_item = label[j].item()
                    triplet_loss_visual_embeddings[label_item].append(anchor_output[j].detach().cpu().numpy())
                    triplet_loss_tactile_embeddings[label_item].append(positive_output[j].detach().cpu().numpy())


        print(f'Epoch {epoch}, Loss: {total_loss/len(triplets_loader)}')

    # Save the embeddings for later use
    np.save('trained_visual_embeddings_MLP.npy', dict(triplet_loss_visual_embeddings))
    np.save('trained_tactile_embeddings_MLP.npy', dict(triplet_loss_tactile_embeddings))

if __name__ == '__main__':
    train_with_triplet_loss(1)