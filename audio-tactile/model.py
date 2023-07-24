import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import random
from torch.utils.data import Dataset

NUM_CLASSES = 7

"""Cross-Entropy Loss Network"""

#Tactile Branch
class ResnetBranch(nn.Module):
    """A network branch based on a pretrained ResNet."""
    def __init__(self, pre_trained=True, hidden_dim=2048, output_dim=200):
        super(ResnetBranch, self).__init__()
        self.base_model = models.resnet50(pretrained=pre_trained)
        self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

        # Enable gradient updates for all layers
        for param in self.base_model.parameters():
            param.requires_grad = True

        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # remove the final fully connected layer
        self.fc = nn.Linear(num_features, output_dim)  # new fc layer for embeddings

    def forward(self, x):
        x = self.base_model(x)
        embeddings = self.fc(x)
        return embeddings

#Audio Branch
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

#Joint Network
class CrossSensoryNetwork(nn.Module):
    def __init__(self, pre_trained=True, hidden_dim=2048, output_dim=200):
        super(CrossSensoryNetwork, self).__init__()
        self.visual_branch = ResnetBranch(pre_trained, hidden_dim, output_dim)
        self.audio_branch = AudioBranch(hidden_dim, output_dim)
        self.joint_fc = nn.Linear(output_dim * 2, NUM_CLASSES)  # to get the joint embeddings
  

    def forward(self, audio_input, visual_input):
        visual_output = self.visual_branch(visual_input)
        audio_output = self.audio_branch(audio_input)
        joint_input = torch.cat((visual_output, audio_output), dim=1)
        joint_embeddings = self.joint_fc(joint_input)
        return audio_output, visual_output, joint_embeddings

"""Triplet Loss Network"""
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 200)

    def forward(self, x):
        if isinstance(x, list):
            # Concatenate the list of embeddings along the batch dimension
            x = torch.cat(x, dim=0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = F.normalize(x, p=2, dim=-1)  # normalize the embeddings to have norm=1
        return x

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class TripletDataset(Dataset):
    def __init__(self, audio_embeddings, tactile_embeddings):
        assert audio_embeddings.keys() == tactile_embeddings.keys()

        self.labels = list(audio_embeddings.keys())
        self.audio_embeddings = audio_embeddings
        self.tactile_embeddings = tactile_embeddings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        positive_source = random.choice(['audio', 'tactile'])

        if positive_source == 'audio':
            positive = random.choice(self.audio_embeddings[label])
            anchor = random.choice(self.tactile_embeddings[label])
        else:
            positive = random.choice(self.tactile_embeddings[label])
            anchor = random.choice(self.audio_embeddings[label])

        while True:
            negative_label = random.choice(self.labels)
            if negative_label != label:
                break

        negative_source = random.choice(['audio', 'tactile'])
        negative = random.choice(self.audio_embeddings[negative_label] if negative_source == 'audio' else self.tactile_embeddings[negative_label])

        return torch.tensor(anchor), torch.tensor(positive), torch.tensor(negative), label