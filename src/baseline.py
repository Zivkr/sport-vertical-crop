from src.dataset import FrameDataset
from src.utils import calculate_iou, calculate_iou_x
from torch.utils.data import Subset
import numpy as np


if __name__ == "__main__":
    fixed_seed = 1337
    dataset = FrameDataset("../data/bounding_boxes.json", "../data/frames")
    splits = np.load(f"../data/splits/split_seed{fixed_seed}.npz")

    train_ds = Subset(dataset, splits["train_idx"].tolist())
    val_ds = Subset(dataset, splits["val_idx"].tolist())
    test_ds = Subset(dataset, splits["test_idx"].tolist())

    n_frames_val = len(val_ds)
    all_squared_error = []
    iou = []
    for idx, frame in enumerate(val_ds):
        image = frame["image"]
        center_x = frame["target"]
        gt_bbox = frame["bbox"]
        image_h, image_w = image.shape[0], image.shape[1]

        # turning ground truth bbox to (x1,y1),(x2,y2)
        gt_bbox_formatted = (gt_bbox[0][0], gt_bbox[0][1], gt_bbox[3][0], gt_bbox[3][1])
        crop_w = image_h * (9 / 16)
        crop_x1 = (image_w / 2) - (crop_w / 2)
        crop_x2 = (image_w / 2) + (crop_w / 2)
        baseline_bbox = (crop_x1, 0, crop_x2, image_h)

        iou_2d = calculate_iou(gt_bbox_formatted, baseline_bbox)
        iou_1d = calculate_iou_x(gt_bbox_formatted, baseline_bbox)
        # our crop is always at the center
        squared_error = (center_x.item() - 0.5) ** 2
        iou.append(iou_2d)
        all_squared_error.append(squared_error)
    print(f"Mean IOU on Validation set (seed {fixed_seed})", np.mean(iou))
    print(f"MSE on Validation set (seed {fixed_seed})", np.mean(all_squared_error))









