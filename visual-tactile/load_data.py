from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder

## Data Loader - modify to select the object numbers, labels, and datset directory.

class CustomDataSet(Dataset):
    def __init__(self, tactile, visual, labels):
        self.tactile = tactile
        self.visual = visual
        self.labels = labels

    def __getitem__(self, index):
        vis = self.visual[index]
        tact = self.tactile[index]
        lab = self.labels[index]
        return tact, vis, lab

    def __len__(self):
        count = len(self.visual)
        assert len(self.visual) == len(self.labels), "Mismatched visual and label lengths."
        return count


def fetch_data():
    TARGET_SIZE = (246, 246)

    tactile_train = []
    tactile_test = []
    visual_train = []
    visual_test = []
    label_train = []
    label_test = []
    
    ## Specify the object numbers to be used for training and testing
    object_numbers = [1, 2, 3, 4, 10, 13, 17, 18, 25, 29, 30, 33, 49, 50, 66, 67, 71, 83, 89, 100]
    # object_numbers = [1, 4, 18, 25, 30, 50, 100]
    # Initialize transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.Resize(TARGET_SIZE),
                                    transforms.ToTensor(),
                                    normalize])

    for object_number in object_numbers:
        folder_dir = f"../../data/tactile/train/{object_number}"
        for images in os.listdir(folder_dir):
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                tactile_train.append(img_tensor)
                label_train.append(object_number)

    for object_number in object_numbers:
        folder_dir = f"../../data/touch/test/{object_number}"
#         count = 0
        for images in os.listdir(folder_dir):
#             if count >= 20:
#                 break
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                tactile_test.append(img_tensor)
                label_test.append(object_number)
#                 count += 1


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
#         count = 0
        for images in os.listdir(folder_dir):
#             if count >= 20:  # Stop if 50 images have been loaded for the current object
#                 break
            # check if the image ends with png
            if (images.endswith(".png")):
                # load the image
                img = Image.open(os.path.join(folder_dir, images))
                # apply transformations
                img_tensor = transform(img)
                visual_test.append(img_tensor)
#                 count += 1


    return tactile_train, tactile_test, visual_train, visual_test, label_train, label_test


def get_loader(batch_size):
    tactile_train, tactile_test, visual_train, visual_test, label_train, label_test = fetch_data()

    ## specify labels depending on number of classes and category labels
    labels = [1, 2, 3, 4, 10, 13, 17, 18, 25, 29, 30, 33, 49, 50, 66, 67, 71, 83, 89, 100]
    # labels = [1, 4, 18, 25, 30, 50, 100]
    encoder = LabelEncoder()
    label_train = encoder.fit_transform(label_train)
    label_test = encoder.fit_transform(label_test)
    #decode the labels: decoded_labels = encoder.inverse_transform(encoded_labels)

    tactile = {'train': tactile_train, 'test': tactile_test}
    visual = {'train': visual_train, 'test': visual_test}
    labels = {'train': label_train, 'test': label_test}
    
    dataset = {x: CustomDataSet(tactile=tactile[x], visual=visual[x], labels=labels[x]) 
               for x in ['train', 'test']}
    
    shuffle = {'train': True, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, 
                                shuffle=shuffle[x], num_workers=0) 
                  for x in ['train', 'test']}

    
    ## get input data parameters - can be used for debugging
    # Assuming 3-dimensional tensors, with dimensions [channel, height, width]
    tactile_dim = tactile_train[0].numel()
    visual_dim = visual_train[0].numel()
    num_class = len(label_train)

    input_data_par = {}
    input_data_par['tactile_test'] = tactile_test
    input_data_par['visual_test'] = visual_test
    input_data_par['label_test'] = label_test
    input_data_par['tactile_train'] = tactile_train
    input_data_par['visual_train'] = visual_train
    input_data_par['label_train'] = label_train
    input_data_par['tactile_dim'] = tactile_dim
    input_data_par['visual_dim'] = visual_dim
    input_data_par['num_class'] = num_class
    
    return dataloader, input_data_par



