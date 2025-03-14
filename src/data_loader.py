import os
import sys
from pathlib import Path
import cv2

# Calculate the project root (assuming this script is in project_root/src/)
project_root = Path(__file__).parent.parent

# Ensure the project root is in sys.path so that config.py can be imported
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import DATASET_DIR, IMAGE_RESOLUTION


def get_sequence_paths(sequence_name):
    """
    Construct and return the paths for:
      - raw JPEG images
      - standard segmentation annotations
      - semantic annotations
    for the given sequence name.
    """
    images_path = project_root / DATASET_DIR / "JPEGImages" / IMAGE_RESOLUTION / sequence_name
    annotations_path = project_root / DATASET_DIR / "Annotations" / IMAGE_RESOLUTION / sequence_name
    semantics_path = project_root / DATASET_DIR / "Annotations_semantics" / IMAGE_RESOLUTION / sequence_name
    return images_path, annotations_path, semantics_path


def load_frames(sequence_name):
    # Get the path to the raw image frames
    images_path, _, _ = get_sequence_paths(sequence_name)
    print("Constructed frames path:", images_path)

    if not images_path.exists():
        raise FileNotFoundError(f"Directory not found: {images_path}")

    # Get a sorted list of image files (filtering for .jpg and .png files)
    frame_files = sorted([file for file in images_path.iterdir() if file.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    frames = [cv2.imread(str(file)) for file in frame_files]
    return frames


if __name__ == "__main__":
    # Replace with a valid sequence name from your ImageSets/2017 train.txt or val.txt
    sequence_name = "dog-agility"  # update to a valid folder name
    images_path, annotations_path, semantics_path = get_sequence_paths(sequence_name)

    print("Images path:", images_path)
    print("Annotations path:", annotations_path)
    print("Semantics path:", semantics_path)

    # Load the frames from the chosen sequence
    frames = load_frames(sequence_name)

    # Display the frames using OpenCV
    for frame in frames:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()