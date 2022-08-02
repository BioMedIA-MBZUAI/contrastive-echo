from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from pl_bolts.models.self_supervised.simclr.transforms import (SimCLREvalDataTransform, SimCLRTrainDataTransform)

import os
import sys
import random
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image

class EchonetSimCLR(torchvision.datasets.VisionDataset):

    def __init__(self, root=None, split="train"):
        self.root = root
        self.split = split
        self.images = {}

        count = 0
        for image in os.listdir(os.path.join(self.root, self.split)):
            video_name = image.split("_")[0]
            if video_name not in self.images.keys():
                self.images[video_name] = []
            self.images[video_name].append(image)
            count += 1

        print("Dataset " + split + " :" + str(len(self.images.keys())))
        print("Dataset " + split + " total:" + str(count))


    def __getitem__(self, index):
        images = self.images[list(self.images.keys())[index]]

        image = random.sample(images, 1)[0]

        img = Image.open(os.path.join(self.root, self.split, image)).convert("RGB")

        if self.split == "train":
            transform = SimCLRTrainDataTransform(224)
        else:
            transform = SimCLREvalDataTransform(224)

        return transform(img), 1

    def __len__(self):
        return len(self.images)


class EchonetSimCLRDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 1

    def setup(self, stage: Optional[str] = None):
        self.data_train = EchonetSimCLR(root=self.data_dir, split="train")
        self.data_val = EchonetSimCLR(root=self.data_dir, split="val")

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
        return None
