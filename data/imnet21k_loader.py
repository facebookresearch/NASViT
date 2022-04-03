from typing import Optional

import torch
from classy_vision.fb.dataset.classy_auto_on_box_everstore_dataset import (
    AutoOnBoxEverstoreDataset,
)
from classy_vision.fb.dataset.transforms import build_transforms

IMAGENET_DEFAULT_MEAN = [0.485, 0.485, 0.485]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

"""
the implementation is adapted from this example
https://www.internalfb.com/intern/diffusion/FBS/browsefile/master/fbcode/mobile-vision/experimental/deit/main.py
"""

HIVE_TABLE_NAMES = {
    # "train": "imagenet_22k_data_migrated_bucketed",
    "train": "imagenet_22k_data_migrated_not_bucketized",
    # Test and Val is not supported in 21K.
}

NUM_SAMPLES = {"train": 14238125}  # Obtained using Diaquery
# label : mobile_vision_workflows/tree/workflows/wbc/zsl/imagenet_meta_data/
# manifold get mobile_vision_dataset/tree/kanchen18/open_images/openimages_sub_training_1m_index.json

HIVE_NAMESPACE = "aml"
EVERSTORE_COLUMN = "everstore_handle"
LABEL_COLUMN = "label"
NUM_CLASSES = 21844  # add n04399382


class ImageNet21KIter:
    """
    A wrapper iterator to make
    ClassyVision loader compatible to the existing torch dataloader.
    ClassyVision loader returns a dictionary while torch loader
    returns a tuple.
    """

    def __init__(self, it):
        self.it = it

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.it)

    def __next__(self):
        obj = self.it.__next__()
        return obj["input"], torch.nonzero(obj["target"])[:, 1]


class ImageNet21KHiveDataset(AutoOnBoxEverstoreDataset):
    """ImageNet-21K dataset."""

    def __init__(
        self,
        split: str,
        batchsize_per_replica: int,
        shuffle: bool,
        transform,
        num_samples: int,
        num_threads_per_worker: int,
        phases_per_epoch: int,
        identity: Optional[str] = None,
        test_mode: bool = False,
    ):
        """Constructor for FBImageNetDataset.
        Args:
            split: only "train" is supported
            batchsize_per_replica: Positive integer indicating batch size for each
                replica
            shuffle: Whether we should shuffle between epochs
            transform: Transform to be applied to each sample
            num_samples: Number of samples in an epoch
            num_threads_per_worker: Number of threads for data loading
            phases_per_epoch: Number of phases to divide the epoch in for
                frequent checkpointing (Default 1)
            identity: Random unique str to identify a dataloader in multi dataset setting
            test_mode: whether we are running in test mode without distributed
                coordination (e.g., Bento notebook)
        """
        hive_config = {
            "namespace": HIVE_NAMESPACE,
            "table_name": HIVE_TABLE_NAMES[split],
            "everstore_column": EVERSTORE_COLUMN,
            "label_column": LABEL_COLUMN,
            "partition_filters": [],
        }
        super().__init__(
            batchsize_per_replica,
            shuffle,
            transform,
            num_samples,
            num_threads_per_worker,
            phases_per_epoch,
            hive_config,
            identity=identity,
            test_mode=test_mode,
            shuffle_memory_limit=self.SHUFFLE_MEMORY_LIMIT_FOR_EMBEDDED_MEDIA,
        )

    @classmethod
    def from_args(cls, batch_size, augment_config=None, input_size=224, num_workers=4, pin_memory=True):
        """Instantiates ImageNet21KHiveDataset.

        Returns:
            An ImageNet21KHiveDataset instance.
        """
        split, shuffle = "train", True

        if augment_config is None:
            augment_config = [
                {"name": "RandomResizedCrop", "size": input_size},
                {"name": "RandomHorizontalFlip"},
                {"name": "ToTensor"},
                {
                    "name": "Normalize",
                    "mean": list(IMAGENET_DEFAULT_MEAN),
                    "std": list(IMAGENET_DEFAULT_STD),
                },
            ]  # Equal to ImagenetAugmentTransform

        # Build a config for transform
        # Use standard IN-1K augmentation.
        transform_config = [
            {
                "name": "image_decode_one_hot",  # image_decode_index
                "config": {
                    "color_space": "RGB",
                    "hashtag_to_class_index_file": "manifold://ondevice_ai_tools/tree/datasets/imagenet21k/class2idx.json",
                    "id_column": "label",
                    "label_column": "label",
                    "img_column": "img",
                    "num_classes": NUM_CLASSES,
                    "single_label": True,
                },
                "transforms": augment_config,
            }
        ]
        transform = build_transforms(transform_config)

        return cls(
            split=split,
            batchsize_per_replica=batch_size,
            shuffle=shuffle,
            transform=transform,
            num_threads_per_worker=num_workers,
            phases_per_epoch=1,
            test_mode=False,
            num_samples=NUM_SAMPLES[split],
        )
