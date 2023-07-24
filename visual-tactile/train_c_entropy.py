from load_data import get_loader

import torch
from torch import nn
from model import CrossSensoryNetwork
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from model import CrossSensoryNetwork

##if training with TPU will need to import these:
# import torch_xla
# import torch_xla.core.xla_model as xm

EPOCHS_C_ENTROPY = 80
BATCH_SIZE = 5

## If training with TPU
# device = xm.xla_device()
## If training with GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_with_cross_entropy(epochs_c_entropy=EPOCHS_C_ENTROPY, batch_size=BATCH_SIZE):

    # Initialize network model
    network = CrossSensoryNetwork().to(device)

    # Initialize your optimizer and loss function
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Get the dataloaders and parameters - parameters for debug only
    dataloader, input_data_par = get_loader(BATCH_SIZE)

    # Get the train and test loaders
    train_loader = dataloader['train']
    test_loader = dataloader['test']

    # Initialize lists to store losses
    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(EPOCHS_C_ENTROPY):
        network.train()  # set network to training mode
        total_train_loss = 0

        # Initialize embeddings storage for each epoch
        visual_embeddings_train = defaultdict(list)
        tactile_embeddings_train = defaultdict(list)

        # Training phase
        for i, (tactile_input, visual_input, targets) in enumerate(train_loader):
            visual_input, tactile_input, targets = visual_input.to(device), tactile_input.to(device), targets.to(device)

            optimizer.zero_grad()

            # Get outputs and embeddings
            visual_output, tactile_output, joint_embeddings = network(visual_input, tactile_input)

            # Compute the loss
            loss = criterion(joint_embeddings, targets)
            total_train_loss += loss.item()

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Save embeddings for each batch
            for j in range(visual_output.shape[0]):
                label = targets[j].item()
                visual_embeddings_train[label].append(visual_output[j].detach().cpu().numpy())
                tactile_embeddings_train[label].append(tactile_output[j].detach().cpu().numpy())

        epoch_train_loss = total_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)  # Append training loss for current epoch

        # Evaluation phase on test set
        network.eval()  # set network to evaluation mode

        # Initialize storage for test embeedings
        visual_embeddings_test = defaultdict(list)
        tactile_embeddings_test = defaultdict(list)
        
        total_test_loss = 0
        
        #Evaluation phase on test set - does not affect the model parameters
        with torch.no_grad():
            for i, (tactile_input, visual_input, targets) in enumerate(test_loader):
                visual_input, tactile_input, targets = visual_input.to(device), tactile_input.to(device), targets.to(device)

                # Get outputs and embeddings
                visual_output, tactile_output, joint_embeddings = network(visual_input, tactile_input)

                # Compute the loss
                loss = criterion(joint_embeddings, targets)
                total_test_loss += loss.item()

                # Save test embeddings for each batch
                for j in range(visual_output.shape[0]):
                    label = targets[j].item()
                    visual_embeddings_test[label].append(visual_output[j].detach().cpu().numpy())
                    tactile_embeddings_test[label].append(tactile_output[j].detach().cpu().numpy())

        test_loss = total_test_loss / len(test_loader)
        test_losses.append(test_loss)  # Append test loss for current epoch
        print(f'Epoch {epoch}, Train Loss: {epoch_train_loss}, Test Loss: {test_loss}')


    # Save the embeddings after all epochs
    print("Training completed. Saving embeddings...")
    np.save('visual_embeddings_kaggle_train.npy', dict(visual_embeddings_train))
    np.save('tactile_embeddings_kaggle_train.npy', dict(tactile_embeddings_train))
    np.save('visual_embeddings_kaggle_test.npy', dict(visual_embeddings_test))
    np.save('tactile_embeddings_kaggle_test.npy', dict(tactile_embeddings_test))

    # Save the trained model
    torch.save(network.state_dict(), 'visual-tactile-model-20.pth')

    # After training, plot the losses
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Train and Test Loss over time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the figure
    plt.savefig("train_test_loss_plot.png")

    # Display the plot
    plt.show()


if __name__ == '__main__':
    train_with_cross_entropy(epochs_c_entropy=EPOCHS_C_ENTROPY, batch_size=BATCH_SIZE)

