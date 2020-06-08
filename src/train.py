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

def run():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_df = pd.read_csv(config.TRAIN_CSV_PATH)
    valid_df = pd.read_csv(config.VALIDATION_CSV_PATH)

    train_dataset = dataset.detection_dataset(train_df, config.IMAGE_DIR, target=config.TARGET_COL, 
                    train=True, transforms=T.Compose([T.ToTensor()]))
    
    valid_dataset = dataset.detection_dataset(valid_df, config.IMAGE_DIR, target=config.TARGET_COL,
                    train=True, transforms=T.Compose([T.ToTensor()]))

    # print(train_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE,
    shuffle=False, collate_fn=utils.collate_fn)

    valid_dataloader = DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False, 
                                    collate_fn=utils.collate_fn)

    print("Data Loaders created")

    detector = model.create_model(config.NUM_CLASSES, backbone=config.BACKBONE)  
    
    params = [p for p in detector.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=config.LEARNING_RATE)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    detector.to(device)

    print("Model loaded to device")

    print("---------------- Training Started --------------")

    for epoch in range(config.EPOCHS):
        loss_value = engine.train_fn(train_dataloader, detector, optimizer, device)
        print("epoch = {}, Training_loss = {}".format(epoch, loss_value))
        # Set the threshold as per needs
        results = engine.eval_fn(valid_dataloader, detector, device, detection_threshold=config.DETECTION_THRESHOLD)
        # Pretty printing the results
        pprint(results)


    # For now just saving one model. I haven't build evaluation metrics which I will use to save best model.

    # torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': detector.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss_value,
    #         }, config.MODEL_SAVE_PATH)

    torch.save(detector.state_dict(), config.MODEL_SAVE_PATH)
    print('-' * 25)
    print("Model Trained and Saved to Disk")


    # print(train_dataloader)
    # images, targets, image_ids = next(iter(train_dataloader))
    # print(images)

    # print(targets)
    # print(image_ids)

if __name__ == "__main__":
    run()