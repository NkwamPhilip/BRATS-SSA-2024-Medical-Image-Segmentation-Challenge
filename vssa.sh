#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=16  
#SBATCH --time=23:00:00
#SBATCH -p compute_full_node
#SBATCH --output=/gpfs/fs0/scratch/u/uanazodo/spark6/v2ssa

module load anaconda3
source activate myenv
python swinv2SSA.py --save_checkpoint --json_list="./jsons/ASNR.json" --distributed --data_dir="/gpfs/fs0/scratch/u/uanazodo/spark6/new/" --max_epochs=300 --val_every=5 --noamp --fold=1 --roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48
#python main.py --resume_ckpt --save_checkpoint --json_list="./jsons/ASNR.json" --data_dir="/gpfs/fs0/scratch/u/uanazodo/spark6/new/" --max_epochs=300 --val_every=5 --noamp --distributed --pretrained_model_name="model.pt" --pretrained_dir="./pretrained_models/" --roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48
#python main.py --json_list="./jsons/ASNR.json" --distributed --data_dir="/gpfs/fs0/scratch/u/uanazodo/spark6/new/" --val_every=5 --noamp --pretrained_model_name="model.pt" --pretrained_dir="./pretrained_models/" --fold=1 --roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48
#python swinv2SSA.py --save_checkpoint --json_list="./jsons/set.json" --data_dir="/gpfs/fs0/scratch/u/uanazodo/spark6/old/" --max_epochs=150 --val_every=5 --noamp --distributed --roi_x=128 --roi_y=128 --roi_z=128  --in_channels=4 --spatial_dims=3 --use_checkpoint --feature_size=48
