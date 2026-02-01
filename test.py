from collections import defaultdict
import torch
from datasets import load_dataset
from ultralytics import YOLO
import os
import cv2
import numpy as np


if __name__ == "__main__":
    output_dir = "./runs/detect/drone-detection-model-test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # Sample images for inference
    images = ["./media/fixed-wing-0.jpg", "./media/quad-0.jpg", "./media/shahed.jpg"]

    # Perform prediction
    results = model.predict(source=images, device=device)

    # Save results
    for result in results:
        result.save(f'{output_dir}/{result.path.split("/")[-1]}')

    # Sample video for inference
    video = "./media/fixed-wing-0.mp4"

    # Open the video capture
    cap = cv2.VideoCapture(video)

    # Get video properties for output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_path = f"{output_dir}/output_detection.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through video frames
    while cap.isOpened():

        # Read a frame
        success, frame = cap.read()

        if not success:
            break

        # Track
        result = model.track(frame, persist=True, verbose=False, device=device)[0]

        # If no boxes detected, write the original frame
        if not result.boxes or not result.boxes.is_track:
            out.write(frame)

            continue

        boxes = result.boxes.xywh.cpu()
        tracks = result.boxes.id.int().cpu().tolist()

        # Display the frame with detections
        frame = result.plot()

        # Plot the tracks
        for box, id in zip(boxes, tracks):
            x, y, w, h = box

            track = track_history[id]
            track.append((float(x), float(y)))

            # Save the last 30 points
            if len(track) > 30:
                track.pop(0)

            # Draw the track line
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                frame, [points], isClosed=False, color=(0, 255, 0), thickness=10
            )

        # Write the frame to output video
        out.write(frame)

    # Release the video capture and close windows
    cap.release()
    out.release()
