import matplotlib.pyplot as plt
import torch
from torch import tensor
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, RandomGrayscale, RandomApply, GaussianBlur, RandomErasing, \
    ColorJitter, Normalize, InterpolationMode

transform = transforms.Compose([
        ToTensor(),
        Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        RandomGrayscale(p=0.12),
        RandomApply([GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.25),
        RandomErasing(p=0.5, scale=(0.02, 0.15)),
        ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
        Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250])),
    ])
val_transform = transforms.Compose([
    ToTensor(),
    Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250])),
])

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
    pos = torch.arange(width, device=centers.device).float()
    # Scale centers from [0,1] to [0, W]
    centers_scaled = centers.unsqueeze(1) * (width - 1)

    # Gaussian formula
    target = torch.exp(-0.5 * ((pos - centers_scaled) / sigma) ** 2)
    target = target / target.sum(dim=1, keepdim=True)
    return target


def format_bboxes(original_shape, gt_bbox, center_pred):
    image_h, image_w = original_shape[0], original_shape[1]

    # turning ground truth bbox to (x1,y1),(x2,y2)
    gt_bbox_formatted = (gt_bbox[0][0], gt_bbox[0][1], gt_bbox[3][0], gt_bbox[3][1])
    crop_w = image_h * (9 / 16)
    crop_x1 = (center_pred * image_w) - (crop_w / 2)
    crop_x2 = (center_pred * image_w) + (crop_w / 2)
    pred_bbox = (crop_x1, 0, crop_x2, image_h)

    return gt_bbox_formatted, pred_bbox


def visualize_confidence(model, val_loader):
    model.eval()
    batch = next(iter(val_loader))
    images = batch["image"]

    # Get prediction AND logits
    pred_x, logits = model(images)
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

    # Plotting
    plt.figure(figsize=(10, 4))

    # 1. The Probability Curve
    plt.plot(probs[0], label='Model Confidence')
    plt.title("Where does the model think the action is?")
    plt.xlabel("Width Position (Left -> Right)")
    plt.ylabel("Probability")

    # 2. The Predicted Center
    # Map 0.0-1.0 pred to 0-28 array index
    idx = pred_x[0].item() * 28
    plt.axvline(x=idx, color='r', linestyle='--', label='Predicted Center')

    plt.legend()
    plt.show()

    # # 2. Flipped Prediction
    # images_flipped = torch.flip(images, dims=[3])  # Flip width dimension
    # pred_flipped = self(images_flipped)
    # pred_flipped_corrected = 1.0 - pred_flipped  # Invert coordinate