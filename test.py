import argparse
import os
from functools import partial

import nibabel as nib
import numpy as np
import torch
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
import json
import math


from monai import data, transforms
from monai.transforms import MapTransform


import numpy as np
from monai.transforms import MapTransform


parser = argparse.ArgumentParser(
    description="Swin UNETR segmentation pipeline")
parser.add_argument("--data_dir", default="",
                    type=str, help="dataset directory")
parser.add_argument("--exp_name", default="segmentation",
                    type=str, help="experiment name")
parser.add_argument("--json_list", default="valid.json",
                    type=str, help="dataset json file")
parser.add_argument("--fold", default=1, type=int, help="data fold")
parser.add_argument("--pretrained_model_name",
                    default="model.pt", type=str, help="pretrained model name")
parser.add_argument("--feature_size", default=48,
                    type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.6, type=float,
                    help="sliding window inference overlap")
parser.add_argument("--in_channels", default=4, type=int,
                    help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int,
                    help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float,
                    help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float,
                    help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float,
                    help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float,
                    help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float,
                    help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float,
                    help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float,
                    help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int,
                    help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int,
                    help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int,
                    help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0,
                    type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true",
                    help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2,
                    type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2,
                    type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1,
                    type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1,
                    type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int,
                    help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true",
                    help="use gradient checkpointing to save memory")
parser.add_argument(
    "--pretrained_dir",
    default="./pretrained_models/",
    type=str,
    help="pretrained checkpoint directory",
)
parser.add_argument("--output_dir", default="",
                    type=str, help="dataset directory")


def create_mri_dataset(input_directory, output_json):
    """
    Create a JSON file listing all BraTS MRI images for inference.

    Args:
        input_directory (str): Path to the directory containing BraTS patient folders
        output_json (str): Path where the output JSON file will be saved
    """
    # Store all patient data
    data_list = []

    # Get all BraTS folders
    brats_folders = [f for f in os.listdir(input_directory)
                     if f.startswith('BraTS-SSA-') and os.path.isdir(os.path.join(input_directory, f))]

    for brats_folder in sorted(brats_folders):  # Sort to process in order
        patient_path = os.path.join(input_directory, brats_folder)

        # Expected file patterns for each sequence
        sequence_patterns = {
            'flair': '-t2f.nii.gz',
            't1ce': '-t1c.nii.gz',
            't1': '-t1n.nii.gz',
            't2': '-t2w.nii.gz'
        }

        # Find sequence files
        sequence_files = {}
        for seq_type, pattern in sequence_patterns.items():
            matching_files = [f for f in os.listdir(
                patient_path) if f.endswith(pattern)]
            if matching_files:
                # Use relative path for portability
                sequence_files[seq_type] = os.path.join(
                    brats_folder, matching_files[0])

        # Check if all sequences are present
        if len(sequence_files) == 4:
            # Create ordered list of sequences
            ordered_paths = [
                sequence_files['flair'],
                sequence_files['t1ce'],
                sequence_files['t1'],
                sequence_files['t2']
            ]

            data_list.append({
                "image": ordered_paths,
            })
        else:
            print(f"Warning: Missing sequences in {brats_folder}")
            print(f"Found sequences: {list(sequence_files.keys())}")

    # Create the final dataset structure
    dataset = {
        "training": data_list
    }

    # Save the result to a JSON file
    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=4)

    # Print summary
    print(f"\nDataset created successfully:")
    print(f"Total patients processed: {len(data_list)}")
    print(f"Dataset saved to: {output_json}")

    # Print first entry as example
    if data_list:
        print("\nExample entry:")
        print(json.dumps(data_list[0], indent=2))


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, shuffle=False):
        """
        Simple sampler for inference that doesn't require distributed processing

        Args:
            dataset: Dataset to sample from
            shuffle: Whether to shuffle the indices (default: False for inference)
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_samples = len(dataset)

    def __iter__(self):
        if self.shuffle:
            # Generate shuffled indices if needed
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            # Sequential indices for standard inference
            indices = list(range(len(self.dataset)))
        return iter(indices)

    def __len__(self):
        return self.num_samples


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    # Process file paths
    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    # Return all files without splitting by fold
    return json_data


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = "valid.json"

    # Get all files without splitting
    all_files = datafold_read(
        datalist=datalist_json, basedir=data_dir, fold=args.fold)

    # Use inference transform
    inference_transform = transforms.Compose(
        [
            transforms.LoadImaged(
                keys=["image"],
                image_only=False
            ),
            transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            transforms.ToTensord(keys=["image"]),
        ]
    )
    # Create dataset with all files
    full_dataset = data.Dataset(data=all_files, transform=inference_transform)

    # Create simplified sampler
    sampler = Sampler(full_dataset, shuffle=False)

    # Create loader
    loader = data.DataLoader(
        full_dataset,
        batch_size=1,
        num_workers=args.workers,
        sampler=sampler,
        pin_memory=True
    )

    return loader


def main():
    args = parser.parse_args()
    create_mri_dataset(args.data_dir, "valid.json")
    args.test_mode = True
    output_directory = args.output_dir + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # [Rest of the main function remains the same]
    pretrained_dir = args.pretrained_dir
    test_loader = get_loader(args)
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    checkpoint = torch.load(pretrained_pth)
    print(f"Checkpoint keys: {checkpoint.keys()}")

    model = SwinUNETR(
        img_size=128,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint
    )

    model_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in model_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"Failed to load state dict: {str(e)}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    model.to(device)

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
    )

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image = batch["image"].cuda()
            affine = batch["image_meta_dict"]["affine"][0].numpy()
            filepath = batch["image_meta_dict"]["filename_or_obj"][0]
            case_id = filepath.split('/')[-2]
            output_name = f"{case_id}.nii.gz"
            print(f"Inference on case {output_name}")

            prob = torch.sigmoid(model_inferer_test(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)

            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 4

            output_path = os.path.join(output_directory, output_name)
            nib.save(nib.Nifti1Image(seg_out.astype(
                np.uint8), affine), output_path)
            print(f"Saved segmentation to {output_path}")

        print("Finished inference!")


if __name__ == "__main__":
    main()
