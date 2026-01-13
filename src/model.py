from src.utils import calculate_iou, generate_gaussian_target, format_bboxes
from src.constants import MOBILENET_MODEL
import lightning as L
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim


class CropModel(L.LightningModule):
    def __init__(self, lr=1e-3, hidden_size=64):
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
        self.conv_head = nn.Sequential(
            nn.Conv2d(int(self.model.num_features), self.hidden_size, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.hidden_size),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.hidden_size, self.hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.hidden_size, self.hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.hidden_size, 1, 1)
        )

        self.save_hyperparameters()

    def forward(self, x):
        image_emb = self.model.forward_features(x)
        heat = self.conv_head(image_emb).squeeze(1)
        x_logits = heat.mean(dim=1)
        p = torch.softmax(x_logits, dim=1)
        pos = torch.linspace(0, 1, p.shape[1], device=x.device)
        x_pred = (p * pos).sum(dim=1, keepdim=True)
        return x_pred, x_logits

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["target"]
        prediction, x_logits = self(images)

        progress = self.current_epoch / self.trainer.max_epochs
        current_sigma = 3.0 - (1.5 * progress)
        target_dist = generate_gaussian_target(labels.squeeze(1), width=x_logits.shape[1], sigma=current_sigma)
        loss = nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(x_logits, dim=1), target_dist)
        self.log('Train Step Loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["target"]
        gt_bboxes = batch["bbox"]
        original_shapes = batch["original_shape"]
        prediction, x_logits = self(images)

        target_dist = generate_gaussian_target(labels.squeeze(1), width=x_logits.shape[1], sigma=1.5)
        loss = nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(x_logits, dim=1), target_dist)

        for gt_bbox, pred, shape in zip(gt_bboxes, prediction, original_shapes):
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
        prediction, _ = self(images)
        loss = self.criterion(prediction, labels)
        for gt_bbox, pred, shape in zip(gt_bboxes, prediction, original_shapes):
            gt_bbox_formatted, pred_bbox = format_bboxes(shape.detach().cpu().numpy(), gt_bbox.detach().cpu().numpy(),
                                                         pred.detach().cpu().item())
            iou = calculate_iou(gt_bbox_formatted, pred_bbox)
            self.test_iou.append(iou)
        return loss

    def on_validation_epoch_end(self):
        mean_iou = np.mean(self.val_iou)
        median_iou = np.median(self.val_iou)
        self.log('Val Mean IOU', float(mean_iou), prog_bar=True)
        self.log('Val Median IOU', float(median_iou), prog_bar=True)
        self.val_iou.clear()

    def on_test_epoch_end(self):
        mean_iou = float(np.mean(self.test_iou))
        median_iou = float(np.median(self.test_iou))
        self.log("Test Mean IOU", mean_iou, prog_bar=True)
        self.log("Test Median IOU", median_iou, prog_bar=True)
        self.test_iou.clear()

    def configure_optimizers(self):
        params = [
            {'params': self.model.parameters(), 'lr': 0.1 * self.lr},
            {'params': self.conv_head.parameters(), 'lr': self.lr}
        ]
        # optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=5e-2)
        optimizer = optim.AdamW(params, lr=self.lr, weight_decay=5e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'Val Step Loss',
        }
