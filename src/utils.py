import torch


def calculate_iou(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2

    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    width = max(inter_x2 - inter_x1, 0)
    height = max(inter_y2 - inter_y1, 0)

    area_inter = width * height
    area_union = (box1_x2 - box1_x1)*(box1_y2 - box1_y1) + (box2_x2 - box2_x1)*(box2_y2 - box2_y1) - area_inter
    return area_inter / area_union


def calculate_iou_x(box1, box2):
    box1_x1, _, box1_x2, _ = box1
    box2_x1, _, box2_x2, _ = box2

    intersect = max(min(box1_x2, box2_x2) - max(box1_x1, box2_x1), 0)
    union = max(box1_x2, box2_x2) - min(box1_x1, box2_x1)
    return intersect / union


def generate_gaussian_target(centers, width, sigma=1.0):
    # centers: [B], width: int, sigma: float
    pos = torch.arange(width, device=centers.device).float()
    # Scale centers from [0,1] to [0, W]
    centers_scaled = centers.unsqueeze(1) * width

    # Gaussian formula
    target = torch.exp(-0.5 * ((pos - centers_scaled) / sigma) ** 2)

    # Normalize to sum to 1 (probability distribution)
    target = target / target.sum(dim=1, keepdim=True)
    return target