from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import librosa

NUM_CLASSES = 7

"""Cross Entropy Network"""

##Visual an Tactile Branches
class ResnetBranch(nn.Module):
    """A network branch based on a pretrained ResNet."""
    def __init__(self, pre_trained=True, hidden_dim=2048, output_dim=200):
        super(ResnetBranch, self).__init__()
        self.base_model = models.resnet50(pretrained=pre_trained)
    
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # remove the final fully connected layer
        self.fc = nn.Linear(num_features, output_dim)  # new fc layer for embeddings

    def forward(self, x):
        x = self.base_model(x)
        embeddings = self.fc(x)
        return embeddings
    
class AudioBranch(nn.Module):
    """A network branch based on 1D CNN for audio data."""
    def __init__(self, hidden_dim=2048, output_dim=200):
        super(AudioBranch, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pooling layer to reduce the size
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # Apply pooling after the convolutions
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x
#Joint Branch
class CrossSensoryNetwork(nn.Module):
    def __init__(self, pre_trained=True, hidden_dim=2048, output_dim=200):
        super(CrossSensoryNetwork, self).__init__()
        self.tactile_branch = ResnetBranch(pre_trained, hidden_dim, output_dim)
        self.audio_branch = AudioBranch(hidden_dim, output_dim)
        self.visual_branch = ResnetBranch(pre_trained, hidden_dim, output_dim)  # new branch
        self.joint_fc = nn.Linear(output_dim * 3, NUM_CLASSES)  # to get the joint embeddings

    def forward(self, audio_input, tactile_input, visual_input):  # new input parameter
        tactile_output = self.tactile_branch(tactile_input)
        audio_output = self.audio_branch(audio_input)
        visual_output = self.visual_branch(visual_input)  # process the visual input
        joint_input = torch.cat((tactile_output, audio_output, visual_output), dim=1)  # concatenate all three
        joint_embeddings = self.joint_fc(joint_input)
        return audio_output, tactile_output, visual_output, joint_embeddings  # return visual output
  
#Pretrain Branch for Tactile
class TactileNetwork(nn.Module):
    def __init__(self, pre_trained=True, hidden_dim=2048, output_dim=200):
        super(TactileNetwork, self).__init__()
        self.tactile_branch = ResnetBranch(pre_trained, hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, NUM_CLASSES)  # final fc layer for classification

    def forward(self, tactile_input):
        tactile_output = self.tactile_branch(tactile_input)
        outputs = self.fc(tactile_output)
        return outputs

## Data Loader - modify to select the object numbers, labels, and datset directory.
class CustomDataSet(Dataset):
    def __init__(self, audio, tactile, visual, labels):
        self.audio = audio
        self.tactile = tactile
        self.visual = visual
        self.labels = labels

    def __getitem__(self, index):
        aud = self.audio[index]
        tac = self.tactile[index]
        vis = self.visual[index]
        lab = self.labels[index]
        return aud, tac, vis, lab

    def __len__(self):
        count = len(self.tactile)
        assert len(self.tactile) == len(self.labels), "Mismatched examples and label lengths."
        return count

def fetch_data():
    TARGET_SIZE = (246, 246)

    audio_train = []
    audio_test = []
    tactile_train = []
    tactile_test = []
    visual_train = []
    visual_test = []
    label_train = []
    label_test = []

    object_numbers = [1, 4, 18, 25, 30, 50, 100]
    
    # Initialize transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.Resize(TARGET_SIZE),
                                    transforms.ToTensor(),
                                    normalize])
    print(torch.cuda.memory_summary())
    for object_number in object_numbers:
        folder_dir = f"../../data/audio/train/{object_number}"
        for audio_files in os.listdir(folder_dir):
            # check if the file ends with wav
            if (audio_files.endswith(".wav")):
                # load the audio
                audio, _ = librosa.load(os.path.join(folder_dir, audio_files), sr=None)
                audio = torch.tensor(audio).unsqueeze(0)  # Add a dimension for the channel
                audio_train.append(torch.tensor(audio))
                print(torch.cuda.memory_usage())
                print()
                label_train.append(object_number)

    for object_number in object_numbers:
        folder_dir = f"../../data/audio/test/{object_number}"
        for audio_files in os.listdir(folder_dir):
            # check if the file ends with wav
            if (audio_files.endswith(".wav")):
                # load the audio
                audio, _ = librosa.load(os.path.join(folder_dir, audio_files), sr=None)
                audio = torch.tensor(audio).unsqueeze(0)  # Add a dimension for the channel
                audio_test.append(torch.tensor(audio))
                label_test.append(object_number)

    for object_number in object_numbers:
        folder_dir = f"../../data/touch/train/{object_number}"
        for images in os.listdir(folder_dir):
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                tactile_train.append(img_tensor)
 
    for object_number in object_numbers:
        folder_dir = f"../../data/touch/test/{object_number}"
        for images in os.listdir(folder_dir):
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                tactile_test.append(img_tensor)

    for object_number in object_numbers:
        folder_dir = f"../../data/vision/train/{object_number}"
        for images in os.listdir(folder_dir):
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                visual_train.append(img_tensor)

    for object_number in object_numbers:
        folder_dir = f"../../data/vision/test/{object_number}"
        for images in os.listdir(folder_dir):
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                visual_test.append(img_tensor)


    return audio_train, audio_test, tactile_train, tactile_test, visual_train, visual_test, label_train, label_test




def get_loader(batch_size):
    audio_train, audio_test, tactile_train, tactile_test, visual_train, visual_test, label_train, label_test = fetch_data()

    encoder = LabelEncoder()
    label_train = encoder.fit_transform(label_train)
    label_test = encoder.fit_transform(label_test)
    
    audio = {'train': audio_train, 'test': audio_test}
    tactile = {'train': tactile_train, 'test': tactile_test}
    visual = {'train': visual_train, 'test': visual_test}
    labels = {'train': label_train, 'test': label_test}
    
    dataset = {x: CustomDataSet(audio=audio[x], tactile=tactile[x], visual=visual[x], labels=labels[x]) 
               for x in ['train', 'test']}
    
    shuffle = {'train': True, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, 
                                shuffle=shuffle[x], num_workers=0) 
                  for x in ['train', 'test']}

    # Assuming 3-dimensional tensors, with dimensions [channel, height, width]
    audio_dim = audio_train[0].numel()
    tactile_dim = tactile_train[0].numel()
    num_class = len(label_train)

    input_data_par = {}
    input_data_par['audio_test'] = audio_test
    input_data_par['tactile_test'] = tactile_test
    input_data_par['label_test'] = label_test
    input_data_par['audio_train'] = audio_train
    input_data_par['tactile_train'] = tactile_train
    input_data_par['label_train'] = label_train
    input_data_par['tactile_dim'] = tactile_dim
    input_data_par['audio_dim'] = audio_dim
    input_data_par['num_class'] = num_class
    
    return dataloader, input_data_par

EPOCHS_PRETRAIN = 15
EPOCHS_C_ENTROPY = 50
BATCH_SIZE = 5

def train_with_cross_entropy(epochs_pre = EPOCHS_PRETRAIN, epochs_cross_entropy=EPOCHS_C_ENTROPY, batch_size=BATCH_SIZE):
    output_dir = '/scratch/users/k21171248/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Initialize list to store losses
    train_losses = []
    test_losses = []

    for epoch in range(EPOCHS_PRETRAIN):
        tactile_network.train()  # set network to training mode
        total_loss = 0

        for i, (_, tactile_input, _, targets) in enumerate(train_loader):
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
            for i, (_, tactile_input, _, targets) in enumerate(test_loader):
                tactile_input, targets = tactile_input.to(device), targets.to(device)
                tactile_output = tactile_network.tactile_branch(tactile_input)
                outputs = tactile_network.fc(tactile_output)
                test_loss = pretrain_criterion(outputs, targets)
                total_test_loss += test_loss.item()

        test_loss = total_test_loss/len(test_loader)
        test_losses.append(test_loss)
        print(f'Pretraining Epoch {epoch}, Test Loss: {test_loss}')

    # Save the model
    torch.save(tactile_network.state_dict(), os.path.join('/scratch/users/k21171248/output','./tactile_network_pretrain.pth'))

    # Plot train and test loss
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Test loss')
    plt.title('Loss Metrics')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()
    # Save the embeddings after pretraining
    print("Pretraining completed. Saving pretrain tactile embeddings...")
    np.save(os.path.join('/scratch/users/k21171248/output','tactile_embeddings_pretrain.npy'), dict(tactile_embeddings_pretrain))
        
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
    for epoch in range(EPOCHS_C_ENTROPY):
        network.train()  # set network to training mode
        total_train_loss = 0

        # Initialize embeddings storage for each epoch
        audio_embeddings_train = defaultdict(list)
        tactile_embeddings_train = defaultdict(list)
        visual_embeddings_train = defaultdict(list)

        # Training phase
        for i, (audio_input, tactile_input, visual_input, targets) in enumerate(train_loader):
            audio_input, tactile_input, visual_input, targets = audio_input.to(device), tactile_input.to(device), visual_input.to(device), targets.to(device)

            optimizer.zero_grad()

            # Get outputs and embeddings
            audio_output, tactile_output, visual_output, joint_embeddings = network(audio_input, tactile_input, visual_input)

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
                visual_embeddings_train[label].append(visual_output[j].detach().cpu().numpy())
        
        epoch_train_loss = total_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)  # Append training loss for current epoch

        # Evaluation phase on test set
        network.eval()  # set network to evaluation mode
        audio_embeddings_test = defaultdict(list)
        tactile_embeddings_test = defaultdict(list)
        visual_embeddings_test = defaultdict(list)
        
        total_test_loss = 0
        with torch.no_grad():
            for i, (audio_input, tactile_input, visual_input, targets) in enumerate(test_loader):
                audio_input, tactile_input, visual_input, targets = audio_input.to(device), tactile_input.to(device), visual_input.to(device), targets.to(device)

                # Get outputs and embeddings
                audio_output, tactile_output, visual_output, joint_embeddings = network(audio_input, tactile_input, visual_input)

                # Compute the loss
                loss = criterion(joint_embeddings, targets)
                total_test_loss += loss.item()

                # Save test embeddings for each batch
                for j in range(audio_output.shape[0]):
                    label = targets[j].item()
                    audio_embeddings_test[label].append(audio_output[j].detach().cpu().numpy())
                    tactile_embeddings_test[label].append(tactile_output[j].detach().cpu().numpy())
                    visual_embeddings_test[label].append(visual_output[j].detach().cpu().numpy())
        
        test_loss = total_test_loss / len(test_loader)
        test_losses.append(test_loss)  # Append test loss for current epoch
        print(f'Epoch {epoch}, Train Loss: {epoch_train_loss}, Test Loss: {test_loss}')

    # Save the embeddings after all epochs
    print("Training completed. Saving embeddings...")
    np.save(os.path.join('/scratch/users/k21171248/output','audio_embeddings_kaggle_train.npy'), dict(audio_embeddings_train))
    np.save(os.path.join('/scratch/users/k21171248/output','tactile_embeddings_kaggle_train.npy'), dict(tactile_embeddings_train))
    np.save(os.path.join('/scratch/users/k21171248/output','visual_embeddings_kaggle_train.npy'), dict(visual_embeddings_train))
    np.save(os.path.join('/scratch/users/k21171248/output','audio_embeddings_kaggle_test.npy'), dict(audio_embeddings_test))
    np.save(os.path.join('/scratch/users/k21171248/output','tactile_embeddings_kaggle_test.npy'), dict(tactile_embeddings_test))
    np.save(os.path.join('/scratch/users/k21171248/output','visual_embeddings_kaggle_test.npy'), dict(visual_embeddings_test))

    # Save the trained model
    torch.save(network.state_dict(), os.path.join('/scratch/users/k21171248/output','audio-tactile-visual-model.pth'))

    # After training, plot the losses
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Train and Test Loss over time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the figure
    plt.savefig(os.path.join('/scratch/users/k21171248/output',"train_test_loss_plot.png"))

    # Display the plot
    plt.show()

if __name__ == '__main__':
    train_with_cross_entropy()
