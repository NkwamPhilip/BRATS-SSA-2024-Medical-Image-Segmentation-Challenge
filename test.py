# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial
import json

import nibabel as nib
import numpy as np
import torch
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
    MapTransform,
)

from monai import data
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR


def datafold_read(datalist, basedir, fold=0, key="validation"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    # tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        # else:
        #     tr.append(d)

    return val


def get_loader(data_dir, json_list, fold):

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            # ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image"])
        ]
    )

    validation_files = datafold_read(
        datalist=json_list, basedir=data_dir, fold=fold)
    val_ds = data.Dataset(data=validation_files, transform=val_transform)

    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return val_loader


def main():
    test_mode = True
    data_dir = "/gpfs/fs0/scratch/u/uanazodo/spark6/spark24/validate"
    json_list = "./jsons/validation_data.json"
    output_directory = "gpfs/fs0/scratch/u/uanazodo/spark6/seg_output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    test_loader = get_loader(data_dir, json_list, fold=0)
    pretrained_pth = "./pretrained_models/model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=True,
    )
    # Load the state dictionary from the pretrained model file
    state_dict = torch.load(pretrained_pth)

    # Extract the actual model state dictionary if wrapped with other info
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove `module.` if present
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    # Load the processed state dictionary into the model
    model.load_state_dict(new_state_dict, strict=False)
    print(new_state_dict)
    print(model)

    model.eval()
    model.to(device)
    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[128, 128, 128],
        sw_batch_size=1,
        predictor=model,
        overlap=0.6,
    )

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image = batch["image"].cuda()
            affine = batch["image_meta_dict"]["original_affine"][0].numpy()
            num = '-'.join(batch["image_meta_dict"]["filename_or_obj"]
                           [0].split("/")[-1].split("/")[-1].split("-")[2:4])
            img_name = "{}.nii.gz".format(num)
            print("Inference on case {}".format(img_name))
            print(image, len(img_name))
            prob = torch.sigmoid(model_inferer_test(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 3
            nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine),
                     os.path.join(output_directory, img_name))
        print("Finished inference!")
        print(output_directory)


if __name__ == "__main__":
    main()
