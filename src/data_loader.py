import os
import sys
from pathlib import Path
import cv2

project_root = Path(__file__).parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import DATASET_DIR, IMAGE_RESOLUTION

def load_frames(sequence_name):

    # Construct the path to the frames folder for the given sequence
    frames_path = Path(os.path.join(project_root, DATASET_DIR, "JPEGImages", IMAGE_RESOLUTION, sequence_name))
    # Debug: print the constructed path
    print("Constructed frames path:", frames_path)
    frame_files = sorted(os.listdir(frames_path))
    frames = [cv2.imread(os.path.join(frames_path, file)) for file in frame_files]
    return frames


if __name__ == "__main__":
    # Test by loading a specific sequence (replace 'example_sequence' with an actual folder name)
    sequence_name = "dog"  # Example sequence name
    frames = load_frames(sequence_name)
    for frame in frames:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()