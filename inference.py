import os
import argparse
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

if __name__ == "__main__":
    # Allow running from command line with custom arguments
    parser = argparse.ArgumentParser(description="YOLO Inference Pipeline")
    parser.add_argument(
        "--url",
        type=str,
        default="https://ultralytics.com/images/bus.jpg",
        help="URL or local path of the image to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Folder to save predicted outputs (default: output)"
    )
    args = parser.parse_args()

    run_inference(args.url, args.output)