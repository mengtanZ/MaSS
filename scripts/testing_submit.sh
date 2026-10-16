#!/usr/bin/env bash
set -e

PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-data}
WEIGHTS=${WEIGHTS:-weights/mass_mix}
MODEL_NAME=mass_mix

# nyu
DATA_PATH="$DATA_ROOT/nyu/testing"
GT_PATH="$DATA_ROOT/nyu/testing/depth"
"$PYTHON" evaluate_depth_submit.py --dataset nyu --eval_split nyu --data_path "$DATA_PATH" --gt_path "$GT_PATH" --model lite-mono --baseline mass --load_weights_folder "$WEIGHTS" --model_name "$MODEL_NAME" --batch_size 6 --num_workers 4 "$@"

# ddad
# DATA_PATH="$DATA_ROOT/ddad/testing"
# GT_PATH="$DATA_ROOT/ddad/testing/depth"
# "$PYTHON" evaluate_depth_submit.py --dataset ddad --eval_split ddad --data_path "$DATA_PATH" --gt_path "$GT_PATH" --model lite-mono --baseline mass --load_weights_folder "$WEIGHTS" --model_name "$MODEL_NAME" --batch_size 6 --num_workers 4 "$@"

# nuscenes
# DATA_PATH="$DATA_ROOT/nuscenes/testing"
# GT_PATH="$DATA_ROOT/nuscenes/testing/depth"
# "$PYTHON" evaluate_depth_submit.py --dataset nuscenes --eval_split nuscenes --data_path "$DATA_PATH" --gt_path "$GT_PATH" --model lite-mono --baseline mass --load_weights_folder "$WEIGHTS" --model_name "$MODEL_NAME" --batch_size 6 --num_workers 4 --gt_height 900 --gt_width 1600 "$@"

# kitti
DATA_PATH="${KITTI_DATA:-$DATA_ROOT/kitti_raw}"
GT_PATH="${KITTI_GT:-$DATA_ROOT/kitti/testing/depth}"
"$PYTHON" evaluate_depth_submit.py --dataset kitti --eval_split eigen --png --data_path "$DATA_PATH" --gt_path "$GT_PATH" --model lite-mono --baseline mass --load_weights_folder "$WEIGHTS" --model_name "$MODEL_NAME" --batch_size 6 --num_workers 4 "$@"

# waymo
# DATA_PATH="$DATA_ROOT/waymo/testing"
# GT_PATH="$DATA_ROOT/waymo/testing/depth"
# "$PYTHON" evaluate_depth_submit.py --dataset waymo --eval_split waymo --data_path "$DATA_PATH" --gt_path "$GT_PATH" --model lite-mono --baseline mass --load_weights_folder "$WEIGHTS" --model_name "$MODEL_NAME" --batch_size 6 --num_workers 4 --gt_height 1280 --gt_width 1920 "$@"

# drivingstereo
# DATA_PATH="$DATA_ROOT/drivingstereo"
# GT_PATH="$DATA_ROOT/drivingstereo/depth-map-full-size"
# "$PYTHON" evaluate_depth_submit.py --dataset drivingstereo --eval_split drivingstereo --data_path "$DATA_PATH" --gt_path "$GT_PATH" --model lite-mono --baseline mass --load_weights_folder "$WEIGHTS" --model_name "$MODEL_NAME" --batch_size 6 --num_workers 4 "$@"

# cityscapes
# DATA_PATH="$DATA_ROOT/cityscapes_eval"
# GT_PATH="${CITYSCAPES_GT:-$DATA_ROOT/cityscapes_eval/gt_depths}"
# "$PYTHON" evaluate_city.py --dataset cityscapes --eval_split cityscapes --data_path "$DATA_PATH" --gt_path "$GT_PATH" --model lite-mono --baseline mass --load_weights_folder "$WEIGHTS" --model_name "$MODEL_NAME" --batch_size 6 --num_workers 4 "$@"

# diml
# DATA_PATH="$DATA_ROOT/diml/testing"
# GT_PATH="$DATA_ROOT/diml/testing/depth"
# "$PYTHON" evaluate_depth_submit.py --dataset diml --eval_split diml --data_path "$DATA_PATH" --gt_path "$GT_PATH" --model lite-mono --baseline mass --load_weights_folder "$WEIGHTS" --model_name "$MODEL_NAME" --batch_size 6 --num_workers 4 "$@"

# ibims
# DATA_PATH="$DATA_ROOT/ibims/testing"
# GT_PATH="$DATA_ROOT/ibims/testing/depth"
# "$PYTHON" evaluate_depth_submit.py --dataset ibims --eval_split ibims --data_path "$DATA_PATH" --gt_path "$GT_PATH" --model lite-mono --baseline mass --load_weights_folder "$WEIGHTS" --model_name "$MODEL_NAME" --batch_size 6 --num_workers 4 "$@"

# sunrgbd
# DATA_PATH="$DATA_ROOT/SUNRGBD/testing"
# GT_PATH="$DATA_ROOT/SUNRGBD/testing/depth"
# "$PYTHON" evaluate_depth_submit.py --dataset sunrgbd --eval_split sunrgbd --data_path "$DATA_PATH" --gt_path "$GT_PATH" --model lite-mono --baseline mass --load_weights_folder "$WEIGHTS" --model_name "$MODEL_NAME" --batch_size 6 --num_workers 4 "$@"

# 7scenes
# DATA_PATH="$DATA_ROOT/7scenes/testing"
# GT_PATH="$DATA_ROOT/7scenes/testing/depth"
# "$PYTHON" evaluate_depth_submit.py --dataset 7scenes --eval_split 7scenes --data_path "$DATA_PATH" --gt_path "$GT_PATH" --model lite-mono --baseline mass --load_weights_folder "$WEIGHTS" --model_name "$MODEL_NAME" --batch_size 6 --num_workers 4 "$@"

# diode
# DATA_PATH="$DATA_ROOT/DIODE/testing"
# GT_PATH="$DATA_ROOT/DIODE/testing/depth"
# "$PYTHON" evaluate_depth_submit.py --dataset diode --eval_split diode --data_path "$DATA_PATH" --gt_path "$GT_PATH" --model lite-mono --baseline mass --load_weights_folder "$WEIGHTS" --model_name "$MODEL_NAME" --batch_size 6 --num_workers 4 "$@"
