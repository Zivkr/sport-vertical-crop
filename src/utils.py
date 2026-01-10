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