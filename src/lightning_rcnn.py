import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
import pytorch_lightning as pl
import config
import model

class lightningRCNN(pl.LightningModule):
    def __init__(self, num_classes, backbone):
        super.__init__()
        self.fastercnn_model = model.create_model(num_classes=num_classes, backbone=backbone)
    
    def forward(self, x):
        out = self.fastercnn_model(x)
    
    def training_step(self, batch, batch_idx):
        xb, yb = batch
        losses = self(xb, list(yb))
        loss_total = sum(losses.values())
        log = {"train/loss": loss, **{f"train/{k}": v for k, v in losses.items()}}
        return {"loss": loss, "log": log}
    
    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        losses = self(xb, list(yb))
        loss_total = sum(losses.values())
        losses = {f"valid/{k}": v for k, v in losses.items()}

        
        # When you want to simply predict
        # self.feature_extractor.eval() 



