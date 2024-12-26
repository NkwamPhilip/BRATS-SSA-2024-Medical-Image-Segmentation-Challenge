import os
import json
from sklearn.model_selection import KFold

def create_mri_dataset(input_directory, output_json, num_folds=5):
    # Store all patient data
    data_list = []

    # Traverse through each patient's folder in the input directory
    for patient_folder in os.listdir(input_directory):
        patient_path = os.path.join(input_directory, patient_folder)

        # Check if the current path is a directory
        if os.path.isdir(patient_path):
            # Extract label from the folder name (adjust the split/parse logic if needed)
            label = patient_folder.split('_')[-1]  # Assumes label is the last part of the folder name after '_'

            # Find MRI images inside each patient folder
            mri_views = []
            for file_name in os.listdir(patient_path):
                file_path = os.path.join(patient_path, file_name)
                if 'flair' in file_name or 't1ce' in file_name or 't1' in file_name or 't2' in file_name:
                    mri_views.append(file_path)

            # Find corresponding label file
            label_file = None
            for file_name in os.listdir(patient_path):
                if file_name.endswith('-seg.nii.gz'):
                    label_file = os.path.join(patient_path, file_name)
                    break

            # Ensure label file is found
            if label_file is None:
                print(f"Warning: No label file found for {patient_folder}. Skipping this folder.")
                continue

            # Add patient data to the list if all required views and label are present
            if len(mri_views) == 4:
                data_list.append({
                    "image": mri_views,
                    "label": label_file
                })

    # Create K-Folds and assign each patient to a fold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    dataset = {"training": []}

    # Assign data to training set for each fold (folds 0 to 4)
    for fold, (_, test_indices) in enumerate(kf.split(data_list)):
        if fold < num_folds:  # Ensure the fold number is within the valid range (0 to 4)
            for idx in test_indices:
                dataset["training"].append({
                    "image": data_list[idx]["image"],
                    "label": data_list[idx]["label"],
                    "fold": fold  # Store fold number as the current fold
                })

    # Save the result to a JSON file
    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f"Dataset with 5 folds (0 to 4) saved to {output_json}")


# Example input directory containing patient folders
input_directory = "/gpfs/fs0/scratch/u/uanazodo/spark6/new/"
output_json = "ASNR.json"  # Output JSON file
create_mri_dataset(input_directory, output_json, num_folds=5)
