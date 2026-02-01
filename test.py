import torch
from ultralytics import YOLO
import os
import cv2


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

    # Loop through video frames
    while cap.isOpened():

        # Read a frame
        success, frame = cap.read()

        if not success:
            break

        # Track
        results = model.predict(frame, conf=0.15, verbose=False, device=device)

        # Display the frame with detections
        annotated_frame = results[0].plot()

        # Write the frame to output video
        out.write(annotated_frame)

    # Release the video capture and close windows
    cap.release()
    out.release()
