import torch
from torch import nn
from model import CrossSensoryNetwork
import numpy as np
from load_data import get_loader
from collections import defaultdict
import matplotlib.pyplot as plt


EPOCHS_C_ENTROPY = 50
BATCH_SIZE = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_with_cross_entropy(epochs_c_entropy=EPOCHS_C_ENTROPY, batch_size=BATCH_SIZE):

    # Get the dataloaders and parameters
    dataloader, input_data_par = get_loader(batch_size)

    # Get the train and test loaders
    train_loader = dataloader['train']
    test_loader = dataloader['test']

    network = CrossSensoryNetwork().to(device)

    # Initialize your optimizer and loss function for the main training
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Initialize lists to store losses
    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(epochs_c_entropy):
        network.train()  # set network to training mode
        total_train_loss = 0

        # Initialize embeddings storage for each epoch
        audio_embeddings_train = defaultdict(list)
        visual_embeddings_train = defaultdict(list)

        # Training phase
        for i, (audio_input, visual_input, targets) in enumerate(train_loader):
            audio_input, visual_input, targets = audio_input.to(device), visual_input.to(device), targets.to(device)

            optimizer.zero_grad()

            # Get outputs and embeddings
            audio_output, visual_output, joint_embeddings = network(audio_input, visual_input)

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
                visual_embeddings_train[label].append(visual_output[j].detach().cpu().numpy())

        epoch_train_loss = total_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)  # Append training loss for current epoch

        # Evaluation phase on test set
        network.eval()  # set network to evaluation mode
        audio_embeddings_test = defaultdict(list)
        visual_embeddings_test = defaultdict(list)
        total_test_loss = 0
        with torch.no_grad():
            for i, (audio_input, visual_input, targets) in enumerate(test_loader):
                audio_input, visual_input, targets = audio_input.to(device), visual_input.to(device), targets.to(device)

                # Get outputs and embeddings
                audio_output, visual_output, joint_embeddings = network(audio_input, visual_input)

                # Compute the loss
                loss = criterion(joint_embeddings, targets)
                total_test_loss += loss.item()

                # Save test embeddings for each batch
                for j in range(audio_output.shape[0]):
                    label = targets[j].item()
                    audio_embeddings_test[label].append(audio_output[j].detach().cpu().numpy())
                    visual_embeddings_test[label].append(visual_output[j].detach().cpu().numpy())

        test_loss = total_test_loss / len(test_loader)
        test_losses.append(test_loss)  # Append test loss for current epoch
        print(f'Epoch {epoch}, Train Loss: {epoch_train_loss}, Test Loss: {test_loss}')

    # Save the embeddings after all epochs
    print("Training completed. Saving embeddings...")
    np.save('audio_embeddings_kaggle_train.npy', dict(audio_embeddings_train))
    np.save('visual_embeddings_kaggle_train.npy', dict(visual_embeddings_train))
    np.save('audio_embeddings_kaggle_test.npy', dict(audio_embeddings_test))
    np.save('visual_embeddings_kaggle_test.npy', dict(visual_embeddings_test))

    # Save the trained model
    torch.save(network.state_dict(), 'audio-visual-model-20.pth')

    # After training, plot the losses
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Train and Test Loss over time', fontsize=18)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)

    # Save the figure
    plt.savefig("train_test_loss_plot.png")

    # Display the plot
    plt.show()


if __name__ == '__main__':
    train_with_cross_entropy()