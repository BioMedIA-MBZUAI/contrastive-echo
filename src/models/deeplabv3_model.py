from typing import Any, List

import numpy as np
import torchvision
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import madgrad
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy

from src.models.simclr_model import SimCLR

class DeepLabV3Model(LightningModule):
    """
    Example of LightningModule for DeepLabV3 Segmentation.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model_name: str = "deeplabv3_resnet50",
        model_path: str = None,
        num_classes: int = 1,
        pretrained: bool = False,
        aux_loss: bool = False,
        pretrained_backbone: bool = True,
        simclr: dict = {"resnet_type": "resnet50", "model_path": None, "num_samples": 0, "batch_size": 256, "temperature": 0.05},
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        lr_scheduler: dict = {"step_period": None},
        optimizer: dict = {"name": "adam"}
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        print(self.hparams)

        model = torchvision.models.segmentation.__dict__[self.hparams.model_name](
                pretrained=self.hparams.pretrained,
                aux_loss=self.hparams.aux_loss,
                pretrained_backbone=self.hparams.pretrained_backbone
        )
        model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, self.hparams.num_classes, kernel_size=model.classifier[-1].kernel_size)

        if self.hparams.model_path != None:
            print("-------Model LOADING: " + self.hparams.model_path + "-------")
            model = self.load_from_checkpoint(self.hparams.model_path)

        elif self.hparams.simclr['model_path'] != None:
            print("-------SimCLR LOADING: " + self.hparams.simclr.model_path + "-------")
            pretext_model = SimCLR(1,
                    num_samples = self.hparams.simclr.num_samples,
                    batch_size = self.hparams.simclr.batch_size,
                    dataset = 'cifar10',
                    temperature=self.hparams.simclr.temperature
            )
            pretext_model.load_state_dict(torch.load(self.hparams.simclr.model_path)["state_dict"])
            model.backbone = torchvision.models._utils.IntermediateLayerGetter(pretext_model.encoder, {'layer4':'out'})

        self.model = model
        print(model)

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.preds = []

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)["out"]
        loss = self.criterion(preds[:, 0, :, :], y[:, 0, :, :])

        return loss, preds, y.type(torch.cuda.LongTensor)

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice", self.dice_coeff(targets, preds), on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice", self.dice_coeff(targets, preds), on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        dice = self.dice_coeff(targets, preds)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/dice", dice, on_step=False, on_epoch=True, prog_bar=True)

        self.preds.append(dice)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        preds = np.array(self.preds)
        std = np.std(preds)
        self.log("test/std", std, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        if self.hparams.lr_scheduler.step_period is None:
            self.hparams.lr_scheduler.step_period = math.inf

        if self.hparams.optimizer.name == "sgd":
            optim = torch.optim.SGD(
                params=self.parameters(), lr=self.hparams.lr, momentum=self.hparams.optimizer.momentum, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer.name == "madgrad":
            optim = madgrad.MADGRAD(
                params=self.parameters(), lr=self.hparams.lr, momentum=self.hparams.optimizer.momentum, weight_decay=self.hparams.weight_decay
            )
        else:
            optim = torch.optim.Adam(
                params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
            )

        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.hparams.lr_scheduler.step_period)

        return [optim], [scheduler]

    def dice_coeff(self, y_true, y_pred):
        inter = np.logical_and(y_pred.detach().cpu().numpy() > 0., y_true.detach().cpu().numpy() > 0.).sum()
        union = np.logical_or(y_pred.detach().cpu().numpy() > 0., y_true.detach().cpu().numpy() > 0.).sum()
        return 2 * inter / (union + inter)
