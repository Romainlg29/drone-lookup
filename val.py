import torch
from ultralytics import YOLO
import os


if __name__ == "__main__":

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(
            f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    # Check for the dataset folder
    dataset_root = "./dataset"
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(
            f"Dataset folder not found at {dataset_root}. Please run train.py to prepare the dataset."
        )

    # Load the best model
    model = YOLO("runs/detect/drone-detection-model/weights/best.pt")

    # Evaluate the model on the validation set
    model.val(name="drone-detection-model-validation", device=device)
