from model import ResnetBranch, CrossSensoryNetwork
from load_data import get_loader
import torch
from torch import nn
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

NUM_CLASSES = 20

EPOCHS_PRETRAIN = 15
EPOCHS_C_ENTROPY = 50
BATCH_SIZE = 5

FILE_SUFFIX = "pre"

class TactileNetwork(nn.Module):
    def __init__(self, pre_trained=True, hidden_dim=2048, output_dim=200):
        super(TactileNetwork, self).__init__()
        self.tactile_branch = ResnetBranch(pre_trained, hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, NUM_CLASSES)  # final fc layer for classification

    def forward(self, tactile_input):
        tactile_output = self.tactile_branch(tactile_input)
        outputs = self.fc(tactile_output)
        return outputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_c_entropy_pretrain(epochs_pretrain = EPOCHS_PRETRAIN, epochs=EPOCHS_C_ENTROPY, batch_size=BATCH_SIZE):
    # Initialize Tactile Network
    tactile_network = TactileNetwork().to(device)

    # Initialize your optimizer and loss function for the pretraining
    pretrain_optimizer = torch.optim.Adam(tactile_network.parameters(), lr=0.001)
    pretrain_criterion = nn.CrossEntropyLoss()

    # Get the dataloaders and parameters
    dataloader, input_data_par = get_loader(batch_size)

    # Get the train and test loaders
    train_loader = dataloader['train']
    test_loader = dataloader['test']

    # Pretraining loop
    tactile_embeddings_pretrain = defaultdict(list)

    # Initialize list to store losses
    train_losses = []
    test_losses = []

    for epoch in range(epochs_pretrain):
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
            
        # End of epoch
        train_loss = total_loss/len(train_loader)
        train_losses.append(train_loss)
        print(f'Pretraining Epoch {epoch}, Train Loss: {train_loss}')

        # Evaluation loop on test set
        tactile_network.eval()  # set network to evaluation mode
        total_test_loss = 0
        with torch.no_grad():
            for i, (_, tactile_input, targets) in enumerate(test_loader):
                tactile_input, targets = tactile_input.to(device), targets.to(device)
                tactile_output = tactile_network.tactile_branch(tactile_input)
                outputs = tactile_network.fc(tactile_output)
                test_loss = pretrain_criterion(outputs, targets)
                total_test_loss += test_loss.item()

        test_loss = total_test_loss/len(test_loader)
        test_losses.append(test_loss)
        print(f'Pretraining Epoch {epoch}, Test Loss: {test_loss}')

    # Save the model
    torch.save(tactile_network.state_dict(), f'./tactile_network_pretrain_{FILE_SUFFIX}.pth')

    # Plot train and test loss
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Test loss')
    plt.title('Loss Metrics', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(f'loss_plot_{FILE_SUFFIX}.png')
    plt.show()
    # Save the embeddings after pretraining
    print("Pretraining completed. Saving pretrain tactile embeddings...")
    np.save(f'tactile_embeddings_pretrain_{FILE_SUFFIX}.npy', dict(tactile_embeddings_pretrain))
        
    network = CrossSensoryNetwork().to(device)

    # Load the pretrained weights into the tactile branch
    network.tactile_branch.load_state_dict(tactile_network.tactile_branch.state_dict())

    # Initialize your optimizer and loss function for the main training
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Initialize lists to store losses
    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(epochs):
        network.train()  # set network to training mode
        total_train_loss = 0

        # Initialize embeddings storage for each epoch
        audio_embeddings_train = defaultdict(list)
        tactile_embeddings_train = defaultdict(list)

        # Training phase
        for i, (audio_input, tactile_input, targets) in enumerate(train_loader):
            audio_input, tactile_input, targets = audio_input.to(device), tactile_input.to(device), targets.to(device)

            optimizer.zero_grad()

            # Get outputs and embeddings
            audio_output, tactile_output, joint_embeddings = network(audio_input, tactile_input)

            # Compute the loss
            loss = criterion(joint_embeddings, targets)
            total_train_loss += loss.item()

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Save embeddings for each batch
            for j in range(audio_output.shape[0]):
                label = targets[j].item()
                audio_embeddings_train[label].append(audio_output[j].detach().cpu().numpy())
                tactile_embeddings_train[label].append(tactile_output[j].detach().cpu().numpy())

        epoch_train_loss = total_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)  # Append training loss for current epoch

        # Evaluation phase on test set
        network.eval()  # set network to evaluation mode
        audio_embeddings_test = defaultdict(list)
        tactile_embeddings_test = defaultdict(list)
        total_test_loss = 0
        with torch.no_grad():
            for i, (audio_input, tactile_input, targets) in enumerate(test_loader):
                audio_input, tactile_input, targets = audio_input.to(device), tactile_input.to(device), targets.to(device)

                # Get outputs and embeddings
                audio_output, tactile_output, joint_embeddings = network(audio_input, tactile_input)

                # Compute the loss
                loss = criterion(joint_embeddings, targets)
                total_test_loss += loss.item()

                # Save test embeddings for each batch
                for j in range(audio_output.shape[0]):
                    label = targets[j].item()
                    audio_embeddings_test[label].append(audio_output[j].detach().cpu().numpy())
                    tactile_embeddings_test[label].append(tactile_output[j].detach().cpu().numpy())

        test_loss = total_test_loss / len(test_loader)
        test_losses.append(test_loss)  # Append test loss for current epoch
        print(f'Epoch {epoch}, Train Loss: {epoch_train_loss}, Test Loss: {test_loss}')

    # Save the embeddings after all epochs
    print("Training completed. Saving embeddings...")
    np.save(f'audio_embeddings_kaggle_train_{FILE_SUFFIX}.npy', dict(audio_embeddings_train))
    np.save(f'tactile_embeddings_kaggle_train_{FILE_SUFFIX}.npy', dict(tactile_embeddings_train))
    np.save(f'audio_embeddings_kaggle_test_{FILE_SUFFIX}.npy', dict(audio_embeddings_test))
    np.save(f'tactile_embeddings_kaggle_test_{FILE_SUFFIX}.npy', dict(tactile_embeddings_test))

    # Save the trained model
    torch.save(network.state_dict(), f'audio-tactile-model-20_{FILE_SUFFIX}.pth')

    # After training, plot the losses
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Train and Test Loss over time', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend(fontsize=16)

    # Save the figure
    plt.savefig(f"train_test_loss_plot_{FILE_SUFFIX}.png")

    # Display the plot
    plt.show()

if __name__ == "__main__":
    train_c_entropy_pretrain()