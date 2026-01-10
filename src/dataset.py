import torch
from torch.utils.data import Dataset
import cv2
import json


class FrameDataset(Dataset):
    def __init__(self, metadata_path, img_dir, transform=None):
        with open(metadata_path, 'r') as f:
            original_data = json.load(f)
        self.data = []
        for f, value in original_data.items():
            all_xs = [point[0] for point in value["bounding_box"]]
            center_x = (min(all_xs) + max(all_xs)) / 2
            self.data.append({
                "filename": f + ".png",
                "bounding_box": value["bounding_box"],
                "video_name": value["video_name"],
                "bbox_center_x": center_x
            })
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = f"{self.img_dir}/{item['filename']}"
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bbox = item['bounding_box']
        center_x = item['bbox_center_x']

        # Normalize target to 0-1 range
        img_h, img_w = image.shape[:2]
        target = torch.tensor([center_x / img_w], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "target": target,
            "bbox": torch.tensor(bbox, dtype=torch.long),
            "original_shape": torch.tensor([img_h, img_w], dtype=torch.long)
        }
