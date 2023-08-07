from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder

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

#     object_numbers = [1, 2, 3, 4, 13, 17, 18, 25, 29, 30, 33, 49, 50, 66, 67, 68, 71, 83, 89, 100]
    object_numbers = [1, 4, 18, 25, 30, 50, 100]
    # Initialize transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.Resize(TARGET_SIZE),
                                    transforms.ToTensor(),
                                    normalize])

    for object_number in object_numbers:
        folder_dir = f"/kaggle/input/audio-visual-data/audio_split/train/{object_number}"
        count = 0
        for audio_files in os.listdir(folder_dir):
            if count >= 280:
                break
            # check if the file ends with wav
            if (audio_files.endswith(".wav")):
                # load the audio
                audio, _ = librosa.load(os.path.join(folder_dir, audio_files), sr=None)
                audio = torch.tensor(audio).unsqueeze(0)  # Add a dimension for the channel
                audio_train.append(torch.tensor(audio))
                label_train.append(object_number)
                count += 1

    for object_number in object_numbers:
        folder_dir = f"/kaggle/input/audio-visual-data/audio_split/test/{object_number}"
        count = 0
        for audio_files in os.listdir(folder_dir):
            if count >= 40:
                break
            # check if the file ends with wav
            if (audio_files.endswith(".wav")):
                # load the audio
                audio, _ = librosa.load(os.path.join(folder_dir, audio_files), sr=None)
                audio = torch.tensor(audio).unsqueeze(0)  # Add a dimension for the channel
                audio_test.append(torch.tensor(audio))
                label_test.append(object_number)
                count += 1

    
    for object_number in object_numbers:
        folder_dir = f"/kaggle/input/my-data/tactile/train/{object_number}"
        count = 0
        for images in os.listdir(folder_dir):
            if count >= 280:
                break
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                tactile_train.append(img_tensor)
                count += 1
 


    for object_number in object_numbers:
        folder_dir = f"/kaggle/input/my-data/tactile/test/{object_number}"
        count = 0
        for images in os.listdir(folder_dir):
            if count >= 40:
                break
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                tactile_test.append(img_tensor)
                count += 1

    for object_number in object_numbers:
        folder_dir = f"/kaggle/input/my-data/visual/train/{object_number}"
        count = 0
        for images in os.listdir(folder_dir):
            if count >= 280:
                break
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                visual_train.append(img_tensor)
                count += 1
 


    for object_number in object_numbers:
        folder_dir = f"/kaggle/input/my-data/visual/test/{object_number}"
        count = 0
        for images in os.listdir(folder_dir):
            if count >= 40:
                break
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                visual_test.append(img_tensor)
                count += 1

    return audio_train, audio_test, tactile_train, tactile_test, visual_train, visual_test, label_train, label_test




def get_loader(batch_size):
    audio_train, audio_test, tactile_train, tactile_test, visual_train, visual_test, label_train, label_test = fetch_data()
#     labels = [1, 2, 3, 4, 13, 17, 18, 25, 29, 30, 33, 49, 50, 66, 67, 68, 71, 83, 89, 100]
    labels = [1, 4, 18, 25, 30, 50, 100]
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