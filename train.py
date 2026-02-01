import torch
from datasets import load_dataset
from ultralytics import YOLO
import os
from typing import TypedDict
from PIL.Image import Image
from tqdm import tqdm
import albumentations as A


class ObjectsData(TypedDict):
    bbox: list[list[float]]
    category: list[int]
    area: list[float]
    id: list[int]


class DroneSample(TypedDict):
    width: int
    height: int
    objects: ObjectsData
    image: Image
    image_id: int


if __name__ == "__main__":

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(
            f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    # Load the drone detection dataset
    dataset = load_dataset(
        "ChinnaSAMY1/drone-detection-dataset", cache_dir="./.datasets"
    )

    print(
        f"Dataset loaded with {len(dataset['train'])} training samples and {len(dataset['test'])} test samples."
    )

    # Check for the dataset folder
    dataset_root = "./dataset"
    if not os.path.exists(dataset_root):

        # Create the dataset folder
        os.makedirs(dataset_root)
        print(f"Created dataset folder at {dataset_root}")

        # Iterate through the dataset splits and save images and annotations
        # The test will be renamed to val later
        for split in ["train", "test"]:

            split_folder = os.path.join(
                dataset_root, "val" if split == "test" else "train"
            )
            os.makedirs(split_folder, exist_ok=True)

            for idx, sample in enumerate(
                tqdm(dataset[split], desc=f"Processing {split}")
            ):
                # Save the image
                image_path = os.path.join(split_folder, f"{idx}.jpg")

                # Type hinting for the sample
                data: DroneSample = sample  # type: ignore

                # Save the image
                data["image"].save(image_path)

                # Save the annotations in YOLO format
                annotation_path = os.path.join(split_folder, f"{idx}.txt")

                with open(annotation_path, "w") as f:

                    # Loop through each object in the sample
                    for bbox, category in zip(
                        data["objects"]["bbox"], data["objects"]["category"]
                    ):

                        # Convert bbox from COCO format [x, y, width, height] to YOLO format [class, x_center, y_center, width, height]
                        x, y, w, h = bbox

                        # Calculate center point
                        x_center = (x + w / 2) / data["width"]
                        y_center = (y + h / 2) / data["height"]

                        # Normalize width and height
                        width = w / data["width"]
                        height = h / data["height"]

                        # Skip invalid bboxes (negative or zero dimensions)
                        if width <= 0 or height <= 0:
                            continue

                        # Write the annotation line
                        f.write(f"{category} {x_center} {y_center} {width} {height}\n")

    # Load the latest 26n YOLO model
    model = YOLO("yolo26n.pt")

    # Training configuration
    conf = {
        "epochs": 10,
        "batch": 8,
        "patience": 5,
        "dropout": 0.2,
        "imgsz": 640,
        "save_period": 1,
        "cache": False,
        "exist_ok": True,
        "plots": True,
    }

    # Augmentations optimized for outdoor high-speed drone detection
    augmentations = A.Compose(
        [
            # Motion blur for high-speed drones
            A.MotionBlur(blur_limit=7, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            # Outdoor lighting conditions
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.6
            ),
            A.HueSaturationValue(p=0.5),
            A.RandomGamma(p=0.4),
            A.RandomShadow(p=0.4),
            # Weather conditions
            A.RandomFog(p=0.2),
            A.RandomSunFlare(p=0.1),
            # Camera compression
            A.ImageCompression(quality_range=(70, 100), p=0.3),
        ]
    )

    # Train the model on the drone detection dataset
    model.train(
        data="conf.yaml",
        **conf,
        device=device,
        name="drone-detection-model",
        augmentations=augmentations,
    )
