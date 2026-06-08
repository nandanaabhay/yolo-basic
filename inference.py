import os
import argparse
import cv2
from ultralytics import YOLO

def run_inference(image_url: str, output_dir: str = "output") -> str:
    """
    Runs YOLO inference on a given image URL (or local path) and saves the
    predicted output image to the specified output folder.

    Args:
        image_url (str): The URL or local path of the input image.
        output_dir (str): The directory where the predicted output will be saved.

    Returns:
        str: The absolute path to the directory containing the saved prediction.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    abs_output_dir = os.path.abspath(output_dir)

    print("Loading pretrained YOLO model...")
    model = YOLO("yolo11n.pt")

    print(f"Running inference on: {image_url}")
    # Run inference and save the output directly in the output_dir.
    # By passing name=".", the results are saved directly in the project directory (abs_output_dir).
    results = model(
        image_url,
        save=True,
        project=abs_output_dir,
        name=".",
        exist_ok=True
    )

    print(f"Inference completed! Predictions saved in: {abs_output_dir}")
    return abs_output_dir

def run_live_detection(model_path: str = "yolo11n.pt", source: str = "0") -> None:
    """
    Captures live video from the specified source (e.g. webcam or video file)
    and performs real-time object detection using YOLO and OpenCV.

    Args:
        model_path (str): Path or name of the YOLO model weights.
        source (str): Video source. Can be a camera index (e.g., '0') or a path to a video file.
    """
    print(f"Loading YOLO model '{model_path}'...")
    model = YOLO(model_path)

    # Convert source to integer if it represents a camera index
    video_source = source
    if source.isdigit():
        video_source = int(source)

    print(f"Opening video source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    window_name = "YOLO Live Object Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Starting live video object detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame or video stream ended.")
            break

        # Run inference on the current frame
        results = model(frame, stream=True)

        for r in results:
            # Draw bounding boxes and labels on the frame
            annotated_frame = r.plot()

            # Display the annotated frame
            cv2.imshow(window_name, annotated_frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Break if the window is closed by the user
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Live video detection stopped.")


if __name__ == "__main__":
    # Allow running from command line with custom arguments
    parser = argparse.ArgumentParser(description="YOLO Inference Pipeline")
    parser.add_argument(
        "--url",
        type=str,
        default="https://ultralytics.com/images/bus.jpg",
        help="URL or local path of the image to process (static image mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Folder to save predicted outputs (static image mode)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live video object detection using webcam"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source for live detection (camera index e.g. '0' or video file path)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLO model path or name (default: yolo11n.pt)"
    )
    args = parser.parse_args()

    if args.live:
        run_live_detection(model_path=args.model, source=args.source)
    else:
        run_inference(args.url, args.output)