import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
from load_data import get_loader
from model import TactileNetwork, CrossSensoryNetwork

EPOCHS_PRETRAIN = 45
EPOCHS_C_ENTROPY = 90
BATCH_SIZE = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def c_entropy_train_with_tactile_pretrain():
    # Initialize your Tactile Network
    tactile_network = TactileNetwork().to(device)

    # Initialize your optimizer and loss function for the pretraining
    pretrain_optimizer = torch.optim.Adam(tactile_network.parameters(), lr=0.001)
    pretrain_criterion = nn.CrossEntropyLoss()

    # Get the dataloaders and parameters
    dataloader, input_data_par = get_loader(BATCH_SIZE)

    # Get the train and test loaders
    train_loader = dataloader['train']
    test_loader = dataloader['test']


    # Pretraining loop
    tactile_embeddings_pretrain = defaultdict(list)

    for epoch in range(EPOCHS_PRETRAIN):
        tactile_network.train()  # set network to training mode
        total_loss = 0

        for i, (_, tactile_input, targets) in enumerate(train_loader):
            tactile_input, targets = tactile_input.to(device), targets.to(device)

            pretrain_optimizer.zero_grad()

            # Get outputs and embeddings
            tactile_output = tactile_network.tactile_branch(tactile_input)
            outputs = tactile_network.fc(tactile_output)

            # Compute the loss
            loss = pretrain_criterion(outputs, targets)
            total_loss += loss.item()

            # Backward and optimize
            loss.backward()
            pretrain_optimizer.step()

            # Save embeddings for each batch
            for j in range(tactile_output.shape[0]):
                label = targets[j].item()
                tactile_embeddings_pretrain[label].append(tactile_output[j].detach().cpu().numpy())

        epoch_loss = total_loss/len(train_loader)
        print(f'Pretraining Epoch {epoch}, Loss: {epoch_loss}')

    # Save the embeddings after pretraining
    print("Pretraining completed. Saving pretrain tactile embeddings...")
    np.save('tactile_embeddings_pretrain.npy', dict(tactile_embeddings_pretrain))


    # Initialize your CrossSensory Network
    network = CrossSensoryNetwork().to(device)

    # Load the pretrained weights into the tactile branch
    network.tactile_branch.load_state_dict(tactile_network.tactile_branch.state_dict())

    # Initialize your optimizer and loss function for the main training
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(EPOCHS_C_ENTROPY):
        network.train()  # set network to training mode
        total_loss = 0

        # Initialize embeddings storage for each epoch
        visual_embeddings = defaultdict(list)
        tactile_embeddings = defaultdict(list)

        for i, (visual_input, tactile_input, targets) in enumerate(train_loader):
            visual_input, tactile_input, targets = visual_input.to(device), tactile_input.to(device), targets.to(device)

            optimizer.zero_grad()

            # Get outputs and embeddings
            visual_output, tactile_output, joint_embeddings = network(visual_input, tactile_input)

            # Compute the loss
            loss = criterion(joint_embeddings, targets)
            total_loss += loss.item()

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Save embeddings for each batch
            for j in range(visual_output.shape[0]):
                label = targets[j].item()
                visual_embeddings[label].append(visual_output[j].detach().cpu().numpy())
                tactile_embeddings[label].append(tactile_output[j].detach().cpu().numpy())

        epoch_loss = total_loss/len(train_loader)
        print(f'Epoch {epoch}, Loss: {epoch_loss}')

    # Save the embeddings after all epochs
    print("Training completed. Saving embeddings...")
    np.save('visual_embeddings_kaggle.npy', dict(visual_embeddings))
    np.save('tactile_embeddings_kaggle.npy', dict(tactile_embeddings))


if __name__ == '__main__':
    c_entropy_train_with_tactile_pretrain()
