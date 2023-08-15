import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import librosa
from torchvision import transforms
import pathlib

CURRENT_DIRECTORY = pathlib.Path(__file__).parent.resolve()
DATASET_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "..", "..", "data")
OBJECT_NUMBERS = [1, 2, 3, 4, 13, 17, 18, 25, 29, 30, 33, 49, 50, 66, 67, 68, 71, 83, 89, 100]

class CustomDataSet(Dataset):
    def __init__(self, audio, visual, labels):
        self.audio = audio
        self.visual = visual
        self.labels = labels

    def __getitem__(self, index):
        vis = self.visual[index]
        aud = self.audio[index]
        lab = self.labels[index]
        return aud, vis, lab

    def __len__(self):
        count = len(self.visual)
        assert len(self.visual) == len(self.labels), "Mismatched visual and label lengths."
        return count

def fetch_data():
    TARGET_SIZE = (246, 246)

    audio_train = []
    audio_test = []
    visual_train = []
    visual_test = []
    label_train = []
    label_test = []
    
    # Initialize transforms for preprocessing
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.Resize(TARGET_SIZE),
                                    transforms.ToTensor(),
                                    normalize])

    for object_number in OBJECT_NUMBERS:
        folder_dir = f"{DATASET_DIRECTORY}/audio/train/{object_number}"
        for audio_files in os.listdir(folder_dir):
            # check if the file ends with wav
            if (audio_files.endswith(".wav")):
                # load the audio
                audio, _ = librosa.load(os.path.join(folder_dir, audio_files), sr=None)
                audio = torch.tensor(audio).unsqueeze(0)  # Add a dimension for the channel
                audio_train.append(audio)
                label_train.append(object_number)
                
    for object_number in OBJECT_NUMBERS:
        folder_dir = f"{DATASET_DIRECTORY}/audio/test/{object_number}"   
        for audio_files in os.listdir(folder_dir):
            # check if the file ends with wav
            if (audio_files.endswith(".wav")):
                # load the audio 
                audio, _ = librosa.load(os.path.join(folder_dir, audio_files), sr=None)
                audio = torch.tensor(audio).unsqueeze(0)  # Add a dimension for the channel
                audio_test.append(torch.tensor(audio))
                label_test.append(object_number)

    for object_number in OBJECT_NUMBERS:
        folder_dir = f"{DATASET_DIRECTORY}/vision/train/{object_number}"
        for images in os.listdir(folder_dir):
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                visual_train.append(img_tensor)
                
    for object_number in OBJECT_NUMBERS:
        folder_dir = f"{DATASET_DIRECTORY}/vision/test/{object_number}"
        for images in os.listdir(folder_dir):
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                visual_test.append(img_tensor)

    return audio_train, audio_test, visual_train, visual_test, label_train, label_test


def get_loader(batch_size):
    audio_train, audio_test, visual_train, visual_test, label_train, label_test = fetch_data()
    
    encoder = LabelEncoder()
    label_train = encoder.fit_transform(label_train)
    label_test = encoder.fit_transform(label_test)
    
    audio = {'train': audio_train, 'test': audio_test}
    visual = {'train': visual_train, 'test': visual_test}
    labels = {'train': label_train, 'test': label_test}
    
    dataset = {x: CustomDataSet(audio=audio[x], visual=visual[x], labels=labels[x]) 
               for x in ['train', 'test']}
    
    shuffle = {'train': True, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, 
                                shuffle=shuffle[x], num_workers=0) 
                  for x in ['train', 'test']}

    ##Parameters for debugging purposes
    # Assuming 3-dimensional tensors, with dimensions [channel, height, width]
    tactile_dim = audio_train[0].numel()
    visual_dim = visual_train[0].numel()
    num_class = len(label_train)

    print(tactile_dim, visual_dim, num_class)
    input_data_par = {}
    input_data_par['tactile_test'] = audio_test
    input_data_par['visual_test'] = visual_test
    input_data_par['label_test'] = label_test
    input_data_par['tactile_train'] = audio_train
    input_data_par['visual_train'] = visual_train
    input_data_par['label_train'] = label_train
    input_data_par['tactile_dim'] = tactile_dim
    input_data_par['visual_dim'] = visual_dim
    input_data_par['num_class'] = num_class
    
    return dataloader, input_data_par
