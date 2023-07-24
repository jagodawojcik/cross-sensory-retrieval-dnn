import torch
from torch import nn
from model import CrossSensoryNetwork
import numpy as np
from load_data import get_loader
from collections import defaultdict

EPOCHS_C_ENTROPY = 3
BATCH_SIZE = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_with_cross_entropy(id, epochs_c_entropy=EPOCHS_C_ENTROPY, batch_size=BATCH_SIZE):

    # Initialize your network
    network = CrossSensoryNetwork().to(device)

    # Initialize your optimizer and loss function
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()


    # Get the dataloaders and parameters
    dataloader, input_data_par = get_loader(batch_size)

    # Get the train and test loaders
    train_loader = dataloader['train']
    test_loader = dataloader['test']

    min_loss = 10000

    # Training loop
    for epoch in range(epochs_c_entropy):
        network.train()  # set network to training mode
        total_loss = 0
        
        # Initialize embeddings storage for each epoch
        audio_embeddings = defaultdict(list)
        visual_embeddings = defaultdict(list)
        
        for i, (audio_input, visual_input, targets) in enumerate(train_loader):
            audio_input, visual_input, targets = audio_input.to(device), visual_input.to(device), targets.to(device)

            optimizer.zero_grad()
            
            # Get outputs and embeddings
            audio_output, visual_output, joint_embeddings = network(audio_input, visual_input)

            # Compute the loss
            loss = criterion(joint_embeddings, targets)
            total_loss += loss.item()

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Save embeddings for each batch
            for j in range(visual_output.shape[0]):
                label = targets[j].item()
                audio_embeddings[label].append(audio_output[j].detach().cpu().numpy())
                visual_embeddings[label].append(visual_output[j].detach().cpu().numpy())

        epoch_loss = total_loss/len(train_loader)
        print(f'Epoch {epoch}, Loss: {epoch_loss}')

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            print("New minimum loss reached, min_loss: {}".format(min_loss), "saving embeddings...")
            # Save the embeddings for later use
            np.save('audio_embeddings_kaggle.npy', dict(audio_embeddings))
            np.save('visual_embeddings_kaggle.npy', dict(visual_embeddings))

train_with_cross_entropy(1)