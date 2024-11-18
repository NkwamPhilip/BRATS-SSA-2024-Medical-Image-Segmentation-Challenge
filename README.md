# MICCAI 2024 Brain Tumor Segmentation Challenge: Team Transformers - Spark6
This repository contains is SWIN-UNETR implementation for brain tumor  segmentation using the  BraTS-GLI Dataset (http://braintumorsegmentation.org/) for pretraining and BraTS-SSA dataset for finetuning and in-house validation.

We employed two different approaches.
(1) Our first approach leverages the large, diverse BraTS-GLI 2021 dataset for initial training to build a strong feature extractor and then adapt the model to the specific characteristics of the BraTS-Africa dataset. The BraTS-GLI dataset includes 1,251 cases, providing a robust foundation for learning general features of glioma segmentation across different institutions.
(2) Train the model exclusively on the BraTS-Africa dataset to focus entirely on African-specific glioma characteristics. This strategy aims to eliminate any potential bias introduced by the BraTS-GLI 2021 dataset and ensures that the model is tailored entirely to the imaging and anatomical characteristics present in African patients.
Our pretrained model was trained on BraTS-GLI with 1251 cases and no additional data. We used a five-fold cross-validation (0-4) with a ratio of 80:20 based on MONAI SwinUNTER implementation.

# Tutorial
A colab file for BraTS21 brain tumor segmentation using Swin UNETR model is provided in the following link.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/3d_segmentation/swin_unetr_brats21_segmentation_3d.ipynb)

Model was trained on compute canada (cc).

# Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

# Data Description
# Pretraining Dataset
Modality: MRI
Size: 1251 3D volumes
Challenge: RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge

# Finetuned Dataset
Modality: MRI
Size: 95 3D volumes (60 Training, 35 Validation)
Challenge: Brain Tumor Segmentation (BraTS) SSA Challenge

For pretraining, json files can be generated using generate.py. Ensure you modify your input data directory in the python file : 
```bash
python generate.py
```

We switched the BraTS-SSA dataset label 3 back to 4, changing blue to yellow

The provided segmentation labels have values of 1(red) for NCR, 2(green) for ED, 4(yellow) for ET, and 0 for everything else.

![image](./assets/fig_brats21.png)



# Models
We provide Swin UNETR models which are pre-trained on BraTS21 dataset as in the following. The folds
correspond to the data split in the [json file](https://developer.download.nvidia.com/assets/Clara/monai/tutorials/brats21_folds.json).

<table>
  <tr>
    <th>Models</th>
    <th>TC</th>
    <th>WT</th>
    <th>ET</th>
    <th>Mean</th>
    <th>Download </th>
  </tr>
<tr>
    <td>SwinUNETR trained on GLI</td>
    <td>0.882</td>
    <td>0.918</td>
    <td>0.833</td>
    <td>0.877</td>
    <td><a href="https://drive.google.com/file/d/15TowXpPZYLqtZdJMymQIiWN511UIYwtD/view?usp=sharing">model</a></td>
</tr>

<tr>
    <td>Swin UNETR with only SSA</td>
    <td>0.557</td>
    <td>0.831</td>
    <td>0.517</td>
    <td>62.1</td>
    <td><a href="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/fold1_f48_ep300_4gpu_dice0_9059.zip">model</a></td>
</tr>

<tr>
    <td>Swin UNETR</td>
    <td>0.663</td>
    <td>0.850</td>
    <td>0.662</td>
    <td>0.725</td>
    <td><a href="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/fold2_f48_ep300_4gpu_dice0_8981.zip">model</a></td>
</tr>

</table>

Mean Dice refers to average Dice of WT, ET and TC tumor semantic classes.

# Training

A Swin UNETR network with standard hyper-parameters for brain tumor semantic segmentation (BraTS dataset) is be defined as:

``` bash
model = SwinUNETR(img_size=(128,128,128),
                  in_channels=4,
                  out_channels=3,
                  feature_size=48,
                  use_checkpoint=True,
                  )
```


The above Swin UNETR model is used for multi-modal MR images (4-channel input) with input image size ```(128, 128, 128)``` and for ```3``` class segmentation outputs and feature size of  ```48```.
More details can be found in [1]. In addition, ```use_checkpoint=True``` enables the use of gradient checkpointing for memory-efficient training.

Using the default values for hyper-parameters, the following command can be used to initiate training using PyTorch native AMP package:
``` bash
python main.py
--feature_size=48
--batch_size=1
--logdir=unetr_test_dir
--fold=0
--optim_lr=1e-4
--lrschedule=warmup_cosine
--infer_overlap=0.5
--save_checkpoint
--val_every=10
--json_list='./jsons/brats21_folds.json'
--data_dir=/brats2021/
--use_checkpoint
--noamp
```

## Training from scratch on single GPU with gradient check-pointing and without AMP

To train a `Swin UNETR` from scratch on a single GPU with gradient check-pointing and without AMP:

```bash
python main.py --json_list=<json-path> --data_dir=<data-path> --val_every=5 --noamp \
--roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48
```

## Training from scratch on multi-GPU with gradient check-pointing and without AMP

To train a `Swin UNETR` from scratch on multi-GPU for 300 epochs with gradient check-pointing and without AMP:

```bash
python main.py --json_list=<json-path> --data_dir=<data-path> --max_epochs=300 --val_every=5 --noamp --distributed \
--roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48
```

## Training from scratch on multi-GPU without gradient check-pointing

To train a `Swin UNETR` from scratch on multi-GPU without gradient check-pointing:

```bash
python main.py --json_list=<json-path> --data_dir=<data-path> --val_every=5 --distributed \
--roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --feature_size=48
```

# Evaluation

To evaluate a `Swin UNETR` on a single GPU, the model path using `pretrained_dir` and model
name using `--pretrained_model_name` need to be provided:

```bash
python test.py --json_list=<json-path> --data_dir=<data-path> --feature_size=<feature-size>\
--infer_overlap=0.6 --pretrained_model_name=<model-name> --pretrained_dir=<model-dir>
```

# Finetuning

Please download the checkpoints for models presented in the above table and place the model checkpoints in `pretrained_models` folder.
Use the following commands for finetuning.

## Finetuning on single GPU with gradient check-pointing and without AMP

To finetune a `Swin UNETR`  model on a single GPU on fold 1 with gradient check-pointing and without amp,
the model path using `pretrained_dir` and model  name using `--pretrained_model_name` need to be provided:

```bash
python main.py --json_list=<json-path> --data_dir=<data-path> --val_every=5 --noamp --pretrained_model_name=<model-name> \
--pretrained_dir=<model-dir> --fold=1 --roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48
```

## Finetuning on multi-GPU with gradient check-pointing and without AMP

To finetune a `Swin UNETR` base model on multi-GPU on fold 1 with gradient check-pointing and without amp,
the model path using `pretrained_dir` and model  name using `--pretrained_model_name` need to be provided:

```bash
python main.py --json_list=<json-path> --distributed --data_dir=<data-path> --val_every=5 --noamp --pretrained_model_name=<model-name> \
--pretrained_dir=<model-dir> --fold=1 --roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48
```

# Segmentation Output

By following the commands for evaluating `Swin UNETR` in the above, `test.py` saves the segmentation outputs
in the original spacing in a new folder based on the name of the experiment which is passed by `--exp_name`.

# Citation
If you find this repository useful, please consider citing UNETR paper:

```
@article{hatamizadeh2022swin,
  title={Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images},
  author={Hatamizadeh, Ali and Nath, Vishwesh and Tang, Yucheng and Yang, Dong and Roth, Holger and Xu, Daguang},
  journal={arXiv preprint arXiv:2201.01266},
  year={2022}
}

@inproceedings{tang2022self,
  title={Self-supervised pre-training of swin transformers for 3d medical image analysis},
  author={Tang, Yucheng and Yang, Dong and Li, Wenqi and Roth, Holger R and Landman, Bennett and Xu, Daguang and Nath, Vishwesh and Hatamizadeh, Ali},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20730--20740},
  year={2022}
}
```

# References
[1]: Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H. and Xu, D., 2022. Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. arXiv preprint arXiv:2201.01266.

[2]: Tang, Y., Yang, D., Li, W., Roth, H.R., Landman, B., Xu, D., Nath, V. and Hatamizadeh, A., 2022. Self-supervised pre-training of swin transformers for 3d medical image analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 20730-20740).

[3] U.Baid, et al., The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification, arXiv:2107.02314, 2021.

[4] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

[6] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q
