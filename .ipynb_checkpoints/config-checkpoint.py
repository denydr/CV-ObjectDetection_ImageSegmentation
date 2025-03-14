import os

# Path to the DAVIS dataset (adjust this to your actual location)
DATASET_DIR = os.path.join("datasets","DAVIS_dataset")

# Choose the resolution:
IMAGE_RESOLUTION = "480p"

# Path to the train and validation text files
TRAIN_LIST = os.path.join("datasets","DAVIS_dataset","ImageSets", "2017", "train.txt")
VAL_LIST = os.path.join("datasets", "DAVIS_dataset", "ImageSets", "2017", "val.txt")

# Inference-related parameters
INFERENCE_BATCH_SIZE = 1  # or as needed for inference

# Hyperparameters (if training or fine-tuning)
#BATCH_SIZE = 4
#NUM_EPOCHS = 20
#LEARNING_RATE = 0.001