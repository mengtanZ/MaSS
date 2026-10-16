#!/usr/bin/env bash
set -e

PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-data}
DINO_WEIGHTS=mass/weights/dinov2_vitb14_reg4_pretrain.pth

DATA_PATH="$DATA_ROOT/ddad/training,$DATA_ROOT/nyu/training,$DATA_ROOT/kitti/training,$DATA_ROOT/nuscenes/training"
"$PYTHON" -m mass.train --num_workers 6 --dataset ddad,nyu,kitti_pp,nuscenes --split ddad,nyu,eigen_zhou_pp,nuscenes --data_path "$DATA_PATH" --model_name mass_mix --model lite-mono --baseline mass --dino_weights "$DINO_WEIGHTS" --num_epochs 10 --batch_size 6 --psedo_depth psedo_depth --contrastive_learning --contrast_weights 0.05 --save_frequency 1 "$@"

# ddad
# DATA_PATH="$DATA_ROOT/ddad/training"
# "$PYTHON" -m mass.train --num_workers 6 --dataset ddad --split ddad --data_path "$DATA_PATH" --model_name mass_ddad --model lite-mono --baseline mass --dino_weights "$DINO_WEIGHTS" --num_epochs 30 --batch_size 6 --psedo_depth psedo_depth --contrastive_learning --contrast_weights 0.05 --save_frequency 10 "$@"

# nyu
# DATA_PATH="$DATA_ROOT/nyu/training"
# "$PYTHON" -m mass.train --num_workers 6 --dataset nyu --split nyu --data_path "$DATA_PATH" --model_name mass_nyu --model lite-mono --baseline mass --dino_weights "$DINO_WEIGHTS" --num_epochs 30 --batch_size 6 --psedo_depth psedo_depth --contrastive_learning --contrast_weights 0.05 --save_frequency 10 "$@"

# kitti
# DATA_PATH="$DATA_ROOT/kitti/training"
# "$PYTHON" -m mass.train --num_workers 6 --dataset kitti_pp --split eigen_zhou_pp --data_path "$DATA_PATH" --model_name mass_kitti --model lite-mono --baseline mass --dino_weights "$DINO_WEIGHTS" --num_epochs 20 --batch_size 6 --psedo_depth psedo_depth --contrastive_learning --contrast_weights 0.05 --save_frequency 5 "$@"

# nuscenes
# DATA_PATH="$DATA_ROOT/nuscenes/training"
# "$PYTHON" -m mass.train --num_workers 6 --dataset nuscenes --split nuscenes --data_path "$DATA_PATH" --model_name mass_nuscenes --model lite-mono --baseline mass --dino_weights "$DINO_WEIGHTS" --num_epochs 10 --batch_size 6 --psedo_depth psedo_depth --contrastive_learning --contrast_weights 0.05 --save_frequency 5 "$@"
