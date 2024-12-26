import os
import shutil
import numpy as np
import nibabel as nib
from tqdm import tqdm

def update_enhancing_tumor_label(input_dir, output_dir):
    """
    Change the enhancing tumor label (3) to yellow (4) in BraTS ASNR datasets.

    Args:
        input_dir (str): Path to the input dataset directory.
        output_dir (str): Path to save the updated segmentation files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for subject_dir in tqdm(os.listdir(input_dir), desc="Processing subjects"):
        subject_path = os.path.join(input_dir, subject_dir)
        if os.path.isdir(subject_path):
            # Locate the segmentation file matching the naming pattern
            seg_file = next((f for f in os.listdir(subject_path)
                            if f.endswith('-seg.nii.gz')), None)
            if seg_file is None:
                print(f"No segmentation file found in {subject_dir}")
                continue

            seg_path = os.path.join(subject_path, seg_file)
            try:
                # Load segmentation data
                seg_img = nib.load(seg_path)
                seg_data = seg_img.get_fdata()

                # Update label 3 (enhancing tumor) to 4 (yellow)
                seg_data[seg_data == 3] = 4

                # Save the updated segmentation file
                updated_subject_dir = os.path.join(output_dir, subject_dir)
                os.makedirs(updated_subject_dir, exist_ok=True)
                out_seg_path = os.path.join(updated_subject_dir, seg_file)
                updated_seg_img = nib.Nifti1Image(
                    seg_data, seg_img.affine, seg_img.header)
                nib.save(updated_seg_img, out_seg_path)

            except Exception as e:
                print(f"Error processing {seg_path}: {e}")

def replace_segmentation_files(source_dir, target_dir):
    """
    Replace segmentation files in target_dir with those from source_dir.

    Args:
        source_dir (str): Path to the directory containing the source segmentation files.
        target_dir (str): Path to the directory where the segmentation files will be replaced.
    """
    for subject_dir in tqdm(os.listdir(target_dir), desc="Processing subjects in target directory"):
        target_subject_path = os.path.join(target_dir, subject_dir)
        source_subject_path = os.path.join(source_dir, subject_dir)

        if os.path.isdir(target_subject_path) and os.path.isdir(source_subject_path):
            # Locate the segmentation file in the source directory
            seg_file = next((f for f in os.listdir(
                source_subject_path) if f.endswith('-seg.nii.gz')), None)
            if seg_file is None:
                print(
                    f"No segmentation file found in source directory for {subject_dir}")
                continue

            # Define full paths for source and target segmentation files
            source_seg_path = os.path.join(source_subject_path, seg_file)
            target_seg_path = os.path.join(target_subject_path, seg_file)

            # Replace the segmentation file in the target directory with the one from the source
            try:
                shutil.copy2(source_seg_path, target_seg_path)
                print(f"Replaced {seg_file} in {subject_dir}")
            except Exception as e:
                print(f"Error replacing {seg_file} in {subject_dir}: {e}")

input_dir = '/gpfs/fs0/scratch/u/uanazodo/spark6/new/'
output_dir = '/gpfs/fs0/scratch/u/uanazodo/spark6/ASNRn/'

update_enhancing_tumor_label(input_dir, output_dir)
replace_segmentation_files(output_dir, input_dir)
