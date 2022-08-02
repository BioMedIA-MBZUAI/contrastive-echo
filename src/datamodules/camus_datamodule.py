from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

import os
import sys
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image

def get_mean_and_std(dataset: torch.utils.data.Dataset,
                     samples: int = 128,
                     batch_size: int = 8,
                     num_workers: int = 4):
    """Computes mean and std from samples from a Pytorch dataset.

    Args:
        dataset (torch.utils.data.Dataset): A Pytorch dataset.
            ``dataset[i][0]'' is expected to be the i-th video in the dataset, which
            should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        samples (int or None, optional): Number of samples to take from dataset. If ``None'', mean and
            standard deviation are computed over all elements.
            Defaults to 128.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 8.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.

    Returns:
       A tuple of the mean and standard deviation. Both are represented as np.array's of dimension (channels,).
    """

    if samples is not None and len(dataset) > samples:
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    mean = 0.
    std = 0.
    for imgs, _ in dataloader:
      imgs = imgs.view(batch_size, imgs.size(1), -1)
      mean += imgs.mean(2).sum(0)
      std += imgs.std(2).sum(0)

    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)

    return mean, std


class Camus(torchvision.datasets.VisionDataset):

    def __init__(self, root=None, split="train", mean=[0, 0, 0], std=[1, 1, 1]):
        self.root = root
        self.mean = mean
        self.std = std
        self.split = split
        self.images = []

        images = {}

        for image in os.listdir(os.path.join(self.root, "imgs", "train")):
            video_name = image.split("_")[0]
            if video_name not in images.keys():
                images[video_name] = []
            images[video_name].append(image)

        images = dict(sorted(images.items()))

        if split == "train":
            keys = list(images.keys())[0:300]
        elif split == "val":
            keys = list(images.keys())[300:350]
        else:
            keys = list(images.keys())[350:400]

        _images = {k: images[k] for k in keys}

        for k in _images:
            self.images.append(_images[k][0])
            self.images.append(_images[k][1])

        print("Dataset " + split + ":" + str(len(self.images)))

    def __getitem__(self, index):
        image = self.images[index]

        img = Image.open(os.path.join(self.root, "imgs", "train", image)).convert("RGB")
        mask = Image.open(os.path.join(self.root, "masks", "train", image)).convert("L")

        thresh = 0
        fn = lambda x : 255 if x > thresh else 0
        mask = mask.convert("L").point(fn, mode='1')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        return transform(img), transforms.ToTensor()(mask)

    def __len__(self):
        return len(self.images)


class CamusDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        data_size: int = 1
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_size = data_size

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 1

    def setup(self, stage: Optional[str] = None):
        mean, std = get_mean_and_std(Camus(root=self.data_dir, split="train"))
        kwargs = {
            "mean": mean,
            "std": std
        }

        self.data_train = Camus(root=self.data_dir, split="train", **kwargs)
        self.data_val = Camus(root=self.data_dir, split="val", **kwargs)
        self.data_test = Camus(root=self.data_dir, split="test", **kwargs)

        if self.data_size < 1:
            num_train_patients = int(np.floor(len(self.data_train) * self.data_size))
            indices = np.random.choice(len(self.data_train), num_train_patients, replace=False)
            self.data_train = torch.utils.data.Subset(self.data_train, indices)

            print("Train Data Size is now: " + str(len(self.data_train)))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
