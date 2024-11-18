import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Define training dataset class
class constructImageDataseTrain(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.dataframe.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name)
        label = self.dataframe.iloc[idx, 6]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define testing dataset class
class constructImageDataseTest(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.dataframe.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name)
        label = self.dataframe.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_data_loaders(batch_size=32):
    """Create and return the data loader for training, validation and testing。"""

    # Read training and test data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # Calculate the training and validation set sizes and split
    train_size = int(0.2 * len(train_df))
    val_size = len(train_df) - train_size
    train_df_split = train_df.sample(n=train_size, random_state=42)
    val_df_split = train_df.drop(train_df_split.index)

    # Create dataset object
    train_dataset = constructImageDataseTrain(train_df_split, 'train', transform)
    val_dataset = constructImageDataseTrain(val_df_split, 'train', transform)
    test_dataset = constructImageDataseTest(test_df, 'test', transform)

    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
