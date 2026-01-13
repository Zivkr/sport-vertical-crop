from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Subset, DataLoader
from src.dataset import FrameDataset
from src.model import CropModel
from src.split_train_test import split_save_to_npz
from src.utils import transform, val_transform
from src.constants import *
import lightning as L
import numpy as np
import torch
import wandb


if __name__ == "__main__":
    dataset_train = FrameDataset("../data/bounding_boxes.json", "../data/frames", is_train=True, transform=transform)
    dataset_val = FrameDataset("../data/bounding_boxes.json", "../data/frames", is_train=False, transform=val_transform)
    # split_save_to_npz(FIXED_SEED)
    splits = np.load(f"../data/splits/split_seed{FIXED_SEED}.npz")

    train_ds = Subset(dataset_train, splits["train_idx"].tolist())
    val_ds = Subset(dataset_val, splits["val_idx"].tolist())
    test_ds = Subset(dataset_val, splits["test_idx"].tolist())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=7, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=7, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=7, persistent_workers=True)

    wandb_logger = WandbLogger(project="smart-crop-vertical", log_model=True, tags=["seed42", "image_embedding","hidden64", "unfrozen", "conv_head", "KL", "heatmap", "transforms"])
    wandb_logger.experiment.config.update({"batch_size": BATCH_SIZE, "epochs": EPOCHS, "seed": FIXED_SEED,
                                           "hidden_size": HIDDEN_SIZE})

    # Training
    model = CropModel(lr=LEARNING_RATE, hidden_size=HIDDEN_SIZE).to(torch.device(DEVICE))
    # model = CropModel.load_from_checkpoint(model_path).to(device)
    checkpoint_callback = ModelCheckpoint(monitor='Val Mean IOU', mode='max', save_top_k=1,
                                          filename="best-{epoch:02d}-{Val Mean IOU:.4f}", save_last=True)
    trainer = L.Trainer(accelerator="mps", max_epochs=EPOCHS, log_every_n_steps=10, logger=wandb_logger,
                        callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")
    wandb.finish()