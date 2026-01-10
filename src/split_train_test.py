import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Subset
from src.dataset import FrameDataset


def grouped_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    n = len(dataset)
    indices = np.arange(n)
    groups = np.array([item["video_name"] for item in dataset.data])

    # test split
    test_gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(test_gss.split(indices, groups=groups))

    # train/val split
    val_size_relative = val_ratio / (train_ratio + val_ratio)
    val_gss = GroupShuffleSplit(n_splits=1, test_size=val_size_relative, random_state=seed)
    train_idx, val_idx = next(val_gss.split(trainval_idx, groups=groups[trainval_idx]))

    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist()), Subset(dataset, test_idx.tolist())


def split_save_to_npz(fixed_seed=42):
    dataset = FrameDataset("../data/bounding_boxes.json", "../data/frames")
    train_ds, val_ds, test_ds = grouped_split(dataset, seed=fixed_seed)
    split_dict = {
        "train_idx": np.array(train_ds.indices),
        "val_idx": np.array(val_ds.indices),
        "test_idx": np.array(test_ds.indices),
    }
    np.savez(f"../data/splits/split_seed{fixed_seed}.npz", **split_dict)
