import dataset
import utils
from pprint import pprint
import config
from torchvision import transforms as T
import pandas as pd
import model
import engine
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from lightning_rcnn import lightningRCNN

def run():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Creating Dataset")

    train_df = pd.read_csv(config.TRAIN_CSV_PATH)
    valid_df = pd.read_csv(config.VALIDATION_CSV_PATH)

    train_dataset = dataset.detection_dataset(train_df, config.IMAGE_DIR, target=config.TARGET_COL, 
                    train=True, transforms=T.Compose([T.ToTensor()]))
    
    valid_dataset = dataset.detection_dataset(valid_df, config.IMAGE_DIR, target=config.TARGET_COL,
                    train=True, transforms=T.Compose([T.ToTensor()]))

    # print(train_dataset)

    print("Dataset Created")

    print("Creating DataLoaders")

    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE,
    shuffle=False, collate_fn=utils.collate_fn)

    valid_dataloader = DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False, 
                                    collate_fn=utils.collate_fn)

    print("Data Loaders created")

    print("Creating Lightning Model")
    faster_rcnn = lightningRCNN(num_classes=config.NUM_CLASSES, backbone=config.BACKBONE)
    print("Model created starting lightning training")

    trainer = pl.Trainer(num_nodes=1, gpus=1, max_epochs=config.EPOCHS)
    trainer.fit(faster_rcnn, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)


if __name__ == "__main__":
    run()