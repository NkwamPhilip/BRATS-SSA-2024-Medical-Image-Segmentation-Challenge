import os
import numpy as np
import nibabel as nib

data_path = "/gpfs/fs0/scratch/u/uanazodo/spark6/spark"

# Check if the data path exists
if not os.path.exists(data_path):
    print(f"Error: Directory {data_path} does not exist.")
else:
    # Iterate over each folder in the data path
    for folder in os.listdir(data_path):
        if 'BraTS' in folder:
            folder_path = os.path.join(data_path, folder)
            # Iterate over each file in the folder
            for file in os.listdir(folder_path):
                if 'seg' in file:
                    file_path = os.path.join(folder_path, file)
                    print(f"Processing file: {file_path}")

                    # Load the segmentation file
                    seg = nib.load(file_path)
                    seg_data = seg.get_fdata()

                    # Replace label 4 with label 3
                    seg_arr = np.array(seg_data)
                    seg_arr[seg_arr == 4] = 3

                    # Save the modified segmentation file
                    new_seg = nib.Nifti1Image(seg_arr, affine=seg.affine)
                    nib.save(new_seg, file_path)

    print("Processing complete!")
