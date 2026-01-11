from albumentations import ImageCompression
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import tensor
from torchvision.transforms import RandomHorizontalFlip, Normalize, ColorJitter, InterpolationMode, ToTensor, \
    Resize, RandomGrayscale, GaussianBlur, RandomApply, RandomAutocontrast, RandomEqualize, RandomAdjustSharpness
from torchvision.transforms.v2 import RandomErasing
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, \
    get_constant_schedule_with_warmup
from src.dataset import FrameDataset
from src.utils import calculate_iou, calculate_iou_x, generate_gaussian_target
from src.constants import MOBILENET_MODEL
from src.split_train_test import split_save_to_npz
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
import lightning as L
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import wandb
import timm


def format_bboxes(original_shape, gt_bbox, center_pred):
    image_h, image_w = original_shape[0], original_shape[1]

    # turning ground truth bbox to (x1,y1),(x2,y2)
    gt_bbox_formatted = (gt_bbox[0][0], gt_bbox[0][1], gt_bbox[3][0], gt_bbox[3][1])
    crop_w = image_h * (9 / 16)
    crop_x1 = (center_pred * image_w) - (crop_w / 2)
    crop_x2 = (center_pred * image_w) + (crop_w / 2)
    pred_bbox = (crop_x1, 0, crop_x2, image_h)

    return gt_bbox_formatted, pred_bbox


class CropModel(L.LightningModule):
    def __init__(self, lr=1e-3, hidden_size=32):
        super().__init__()
        self.lr = lr
        self.hidden_size = hidden_size
        self.criterion = nn.SmoothL1Loss()
        self.val_iou = []
        self.test_iou = []

        self.model = timm.create_model(MOBILENET_MODEL, pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = True
        self.original_hidden_size = int(self.model.head_hidden_size)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.model.forward_features(dummy_input)
            # features shape will be [1, 960, 7, 7]
            # self.flatten_dim :int = features.shape[1] * features.shape[2] * features.shape[3]
            self.flatten_dim: int = self.hidden_size * features.shape[2] * features.shape[3]
        self.conv_head = nn.Sequential(
            nn.Conv2d(int(self.model.num_features), self.hidden_size, kernel_size=(1, 1)),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(self.hidden_size, 1, 1)
        )

        self.save_hyperparameters()

    def forward(self, x):
        image_emb = self.model.forward_features(x)
        heat = self.conv_head(image_emb).squeeze(1)  # [B,H,W]
        x_logits = heat.mean(dim=1)  # [B,W]
        p = torch.softmax(x_logits, dim=1)  # [B,W]
        pos = torch.linspace(0, 1, p.shape[1], device=x.device)
        x_pred = (p * pos).sum(dim=1, keepdim=True)
        return x_pred
        # return self.classifier(self.conv_head(image_emb))

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        label = batch["target"]
        prediction = self(images)
        loss = self.criterion(prediction, label)
        self.log('Train Step Loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["target"]
        gt_bboxes = batch["bbox"]
        original_shapes = batch["original_shape"]
        prediction = self(images)
        loss = self.criterion(prediction, labels)
        for gt_bbox, pred, shape in zip(gt_bboxes, prediction, original_shapes):
            # gt_bbox_formatted, pred_bbox = format_bboxes(image, gt_bbox.detach().cpu().numpy(), pred.detach().cpu().item())
            gt_bbox_formatted, pred_bbox = format_bboxes(shape.detach().cpu().numpy(), gt_bbox.detach().cpu().numpy(), pred.detach().cpu().item())
            iou = calculate_iou(gt_bbox_formatted, pred_bbox)
            self.val_iou.append(iou)
        self.log('Val Step Loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["target"]
        gt_bboxes = batch["bbox"]
        original_shapes = batch["original_shape"]
        prediction = self(images)
        loss = self.criterion(prediction, labels)
        for gt_bbox, pred, shape in zip(gt_bboxes, prediction, original_shapes):
            # gt_bbox_formatted, pred_bbox = format_bboxes(image, gt_bbox.detach().cpu().numpy(), pred.detach().cpu().item())
            gt_bbox_formatted, pred_bbox = format_bboxes(shape.detach().cpu().numpy(), gt_bbox.detach().cpu().numpy(),
                                                         pred.detach().cpu().item())
            iou = calculate_iou(gt_bbox_formatted, pred_bbox)
            self.test_iou.append(iou)
        # self.log('test_loss', loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        mean_iou = np.mean(self.val_iou)
        self.log('Val Mean IOU', float(mean_iou), prog_bar=True)
        self.val_iou.clear()

    def on_test_epoch_end(self):
        mean_iou = float(np.mean(self.test_iou)) if len(self.test_iou) else 0.0
        self.log("Test Mean IOU", mean_iou, prog_bar=True)
        self.test_iou.clear()

    def configure_optimizers(self):
        params = [
            {'params': self.model.parameters(), 'lr': 1e-3},
            {'params': self.conv_head.parameters(), 'lr': 1e-2}
        ]
        # optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=5e-2)
        optimizer = optim.AdamW(params, lr=self.lr, weight_decay=5e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5
        )
        # scheduler = get_cosine_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=20,
        #     num_training_steps=self.trainer.estimated_stepping_batches,
        # )
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         "scheduler": scheduler,
        #         'interval': 'step'
        #     }
        # }
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'Val Step Loss',
        }

if __name__ == "__main__":
    fixed_seed = 1701
    batch_size = 32
    learning_rate = 1e-3
    epochs = 20
    hidden_size = 128
    device = torch.device("mps")

    transform = transforms.Compose([
        ToTensor(),
        Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        # RandomGrayscale(p=0.12),
        # RandomApply([GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.25),
        # RandomErasing(p=0.5, scale=(0.02, 0.15)),
        ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
        Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250])),
    ])
    val_transform = transforms.Compose([
        ToTensor(),
        Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250])),
    ])
    dataset_train = FrameDataset("../data/bounding_boxes.json", "../data/frames", is_train=True, transform=transform)
    dataset_val = FrameDataset("../data/bounding_boxes.json", "../data/frames", is_train=False, transform=val_transform)
    # split_save_to_npz(fixed_seed)
    splits = np.load(f"../data/splits/split_seed{fixed_seed}.npz")

    train_ds = Subset(dataset_train, splits["train_idx"].tolist())
    val_ds = Subset(dataset_val, splits["val_idx"].tolist())
    test_ds = Subset(dataset_val, splits["test_idx"].tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=7, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=7, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=7, persistent_workers=True)

    wandb_logger = WandbLogger(project="smart-crop-vertical", log_model="all", tags=["seed42", "image_embedding","hidden128", "unfrozen", "conv_head", "L1Loss", "heatmap", "transforms"])
    wandb_logger.experiment.config.update({"batch_size": batch_size, "epochs": epochs})

    # Training
    model = CropModel(lr=learning_rate, hidden_size=hidden_size).to(device)
    checkpoint_callback = ModelCheckpoint(monitor='Val Step Loss', mode='min', save_top_k=1,
                                          filename="best-{epoch:02d}-{val_loss:.4f}", save_last=True)
    trainer = L.Trainer(accelerator="mps", max_epochs=epochs, log_every_n_steps=10, logger=wandb_logger,
                        callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")
    wandb.finish()