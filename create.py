import os
import json

# Base directories containing the dataset
# Replace with your training data directory
TRAIN_DATA_DIR = "/gpfs/fs0/scratch/u/uanazodo/spark6/spark"
OUTPUT_FILE = "/gpfs/fs0/scratch/u/uanazodo/spark6/BRATS24/jsons/new21.json"

# Initialize the JSON structure
data = {"training": []}


def extract_modality_order(filename):
    """ Extract modality order based on file name format. """
    try:
        # Extract the modality order from the filename
        parts = filename.split('_')
        return parts[2] if len(parts) > 2 else filename
    except IndexError:
        return filename


def process_directory(dir_path, fold_start, data_type):
    print(f"Processing directory: {dir_path}")  # Debug line

    fold = fold_start

    for case in os.listdir(dir_path):
        case_path = os.path.join(dir_path, case)
        if os.path.isdir(case_path):
            images = []
            label = ""

            # Iterate over the files in the case directory
            for file_name in os.listdir(case_path):
                if file_name.endswith(".nii.gz"):
                    file_path = os.path.join(case_path, file_name)
                    if "seg" in file_name:
                        label = file_path
                    else:
                        images.append(file_path)

            # Sort the image files by modality order (flair, t1ce, t1, t2)
            sorted_images = sorted(
                images, key=lambda x: extract_modality_order(os.path.basename(x)))

            # Construct the JSON object for this case
            json_object = {
                "fold": fold,
                "image": sorted_images,
                "label": label
            }

            # Append the JSON object to the training array
            data[data_type].append(json_object)

            fold += 1  # Increment fold number


# Process the training directory
process_directory(TRAIN_DATA_DIR, 0, "training")

# Write the JSON data to the output file
with open(OUTPUT_FILE, 'w') as f:
    json.dump(data, f, indent=4)

print(f"JSON file created at: {OUTPUT_FILE}")
