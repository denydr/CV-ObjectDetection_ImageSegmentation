import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path
import data_loader  # Your module for loading frames

# Configure the tracker parameters as needed.
class Tracker:
    def __init__(self, max_age=30, n_init=3, max_cosine_distance=0.2):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, max_cosine_distance=max_cosine_distance)

    def update(self, boxes, scores, class_ids, frame):
        # Prepare detections in the expected format.
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            detections.append((box, score, class_id))
        # Update the tracker with the current detections.
        tracks = self.tracker.update_tracks(detections, frame=frame)
        output_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            # Instead of storing the track_id, only store the bounding box and detection class.
            bbox = track.to_ltrb()  # format: [x1, y1, x2, y2]
            det_class = track.get_det_class()  # the detected class for the track
            output_tracks.append((bbox, det_class))
        return output_tracks

# Example usage for testing purposes:
if __name__ == "__main__":
    # Use one of your sequences; for example, load raw frames from the "bike-packing" sequence.
    sequence_name = "bike-packing"
    raw_frames = data_loader.load_raw_frames(sequence_name)
    print(f"Loaded {len(raw_frames)} frames for sequence '{sequence_name}'.")

    tracker = Tracker()

    # Create a display window.
    window_name = "Tracking on Sequence"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for frame_filename, frame in raw_frames:
        # For demonstration purposes, use fixed dummy detections.
        # In your actual pipeline, replace these with your model's per-frame detections.
        boxes = np.array([[50, 50, 200, 200], [300, 100, 450, 250]])
        scores = np.array([0.9, 0.85])
        class_ids = np.array([0, 3])

        tracks = tracker.update(boxes, scores, class_ids, frame)
        # Draw the tracking results on the frame.
        annotated_frame = frame.copy()
        for bbox, _ in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # No cv2.putText is called, so no track IDs are drawn.

        # Show the annotated frame.
        cv2.imshow(window_name, annotated_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()