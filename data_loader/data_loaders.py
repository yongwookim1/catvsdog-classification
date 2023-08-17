import os

from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
from PIL import Image


class CatandDogDataset(Dataset):
    def __init__(self, filenames, transforms):
        self.filenames = filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        img = Image.open(img_path)
        img = np.array(img)
        transformed_img = self.transforms(image=img)["image"]
        label = 1 if "cat" in img_path else 0
        return transformed_img, label


class CatandDogInferenceDataset(Dataset):
    def __init__(self, filenames, transforms):
        self.filenames = filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        img = Image.open(img_path)
        img = np.array(img)
        transformed_img = self.transforms(image=img)["image"]
        img_name = img_path.split("/")[-1]
        return transformed_img, img_name


def data_loader(type):
    cat_train_dir = "/opt/ml/data/training_set/training_set/cats"
    dog_train_dir = "/opt/ml/data/training_set/training_set/dogs"
    cat_valid_dir = "/opt/ml/data/valid_set/valid_set/cats"
    dog_valid_dir = "/opt/ml/data/valid_set/valid_set/dogs"
    test_dir = "/opt/ml/data/test_img"

    cat_train_filenames = sorted(
        [os.path.join(cat_train_dir, f) for f in os.listdir(cat_train_dir)]
    )
    dog_train_filenames = sorted(
        [os.path.join(dog_train_dir, f) for f in os.listdir(dog_train_dir)]
    )
    cat_valid_filenames = sorted(
        [os.path.join(cat_valid_dir, f) for f in os.listdir(cat_valid_dir)]
    )
    dog_valid_filenames = sorted(
        [os.path.join(dog_valid_dir, f) for f in os.listdir(dog_valid_dir)]
    )
    test_images_filenames = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
    train_images_filenames = [*cat_train_filenames, *dog_train_filenames]
    valid_images_filenames = [*cat_valid_filenames, *dog_valid_filenames]

    train_transforms = A.Compose(
        [
            A.Resize(height=256, width=256),
            A.HorizontalFlip(p=0.5),
            A.CLAHE(
                always_apply=False, p=1.0, clip_limit=(4, 4), tile_grid_size=(8, 8)
            ),
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=255.0,
                always_apply=True,
            ),
            ToTensorV2(always_apply=True),
        ]
    )
    valid_transforms = A.Compose(
        [
            A.Resize(height=256, width=256),
            A.CLAHE(
                always_apply=False, p=1.0, clip_limit=(4, 4), tile_grid_size=(8, 8)
            ),
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=255.0,
                always_apply=True,
            ),
            ToTensorV2(always_apply=True),
        ]
    )
    test_transforms = A.Compose(
        [
            A.Resize(height=256, width=256),
            A.CLAHE(
                always_apply=False, p=1.0, clip_limit=(4, 4), tile_grid_size=(8, 8)
            ),
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=255.0,
                always_apply=True,
            ),
            ToTensorV2(always_apply=True),
        ]
    )

    train_dataset = CatandDogDataset(train_images_filenames, train_transforms)
    valid_dataset = CatandDogDataset(valid_images_filenames, valid_transforms)
    test_dataset = CatandDogInferenceDataset(test_images_filenames, test_transforms)

    if type == "train":
        train_loader = DataLoader(
            train_dataset, batch_size=64, num_workers=4, shuffle=True, drop_last=True
        )
        return train_loader

    if type == "valid":
        valid_loader = DataLoader(
            valid_dataset, batch_size=64, num_workers=4, shuffle=False, drop_last=True
        )
        return valid_loader
    if type == "test":
        test_loader = DataLoader(
            test_dataset, batch_size=1, num_workers=4, shuffle=False, drop_last=False
        )
        return test_loader
