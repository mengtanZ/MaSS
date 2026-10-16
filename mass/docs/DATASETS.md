# Dataset preparation and evaluation protocols

Use the supplied split files exactly. Training uses preprocessed monocular sequences; testing uses the prepared evaluation layouts below. This repository does not contain raw-data conversion pipelines, pseudo-depth generation, datasets or ground truth. Acquire the datasets from their respective providers under their terms, then prepare the required layouts. Existing preprocessed MaSS data can be used directly.

## Training layout

The four datasets use `mass/splits/{ddad,nyu,eigen_zhou_pp,nuscenes}/{train,val}_files.txt`. Each entry is `relative_sequence frame_index side`. The final side token may be a placeholder for preprocessed data. For example:

```text
DATA_ROOT/ddad/training/
  sequence/
    cam.txt
    000123.jpg
    000122.jpg
    000124.jpg
    psedo_depth/
      000123.png
```

DDAD, NYUv2 and nuScenes training use six-digit RGB/pseudo-depth filenames. KITTI preprocessed training uses ten-digit filenames under each listed folder, for example `0000000123.jpg` and `psedo_depth/0000000123.png`. All training recipes default to JPG input; use `--png` only if every selected training dataset uses PNG.

`cam.txt` is a whitespace-delimited 3x3 pixel camera matrix at the stored RGB resolution. The loader normalizes it using the actual source image dimensions. KITTI requires this file. The generic loader retains the original fallback normalized intrinsics when it is absent; supply actual intrinsics for calibrated reproduction.

Pseudo-depth maps can be prepared Depth Anything V2 (Metric) outputs encoded as PNG in the existing `psedo_depth` representation. Preserve the numerical encoding and scale of the original preprocessing: the structural loss divides these values by 500. Arbitrarily rescaling pseudo-depth images changes that objective. Raw DAV2 float outputs need conversion to this representation before training. This release does not supply a new pseudo-depth generation recipe or claim it reproduces the original preprocessing from raw datasets.

The temporal inputs are frame IDs `[0, -1, 1]`. The original loader's behavior of substituting the target image for a missing neighbor is retained. Mixed training concatenates the four training splits and samples uniformly over the resulting combined image list; it does not balance dataset sizes.

## Evaluation layouts

Configure `DATA_ROOT` and edit the path variables directly in `scripts/training_submit.sh` and `scripts/testing_submit.sh`. Generic evaluation entries are `relative_folder frame_index side`; filenames below are appended to that folder. Ground-truth files are sorted lexicographically and must correspond one-to-one with split entries in their existing order. Count checks detect missing/extra files but cannot establish semantic image/GT correspondence; retain original preparation and naming.

| Dataset | Split | RGB filename/layout | Ground truth | Valid depth (m) |
| --- | --- | --- | --- | --- |
| KITTI | `eigen` | Raw KITTI tree: `date/drive/image_02/data/0000000000.jpg` (or PNG with `--png`) | NPY depth maps, `KITTI_GT` | 0.001–80 |
| DDAD | `ddad` | `000000.jpg` | SciPy sparse NPZ matrices | 0.001–200 |
| NYUv2 | `nyu` | `0000.png` | PNG / 5000 | 0.001–10 |
| nuScenes | `nuscenes` | `000000.jpg` | NPY rows `(x_pixel,y_pixel,depth_m)`; 900x1600 | 0.001–75 |
| Waymo | `waymo` | `000000.jpg` | NPY rows `(x_pixel,y_pixel,depth_m)`; 1280x1920 | 0.001–75 |
| DrivingStereo | `drivingstereo` | `000000.png` in split-relative folder | PNG / 256, `DRIVINGSTEREO_GT` | 0.001–75 |
| Cityscapes | `cityscapes` | `city/frame_leftImg8bit.png`; split entry `city frame` | `000_depth.npy`, `001_depth.npy`, …, `CITYSCAPES_GT` | 0.001–80 |
| DIML | `diml` | `0000.png` | PNG / 1000 | 0.1–10 |
| iBims-1 | `ibims` | `0000.png` | PNG / 1000 | 0.001–10 |
| SUN-RGBD | `sunrgbd` | `000000.jpg` | PNG / 5000 | 0.1–10 |
| 7-Scenes | `7scenes` | `000000.png` | PNG / 5000 | 0.1–10 |
| DIODE | `diode` | `000000.png` | NPY depth in metres and sorted NPY masks under `<data_path>/mask` | 0.001–75 |

Unless overridden, GT lives at `<data_path>/depth`; the script sets explicit overrides for KITTI, DrivingStereo and Cityscapes. Cityscapes also needs `camera/test/<city>/<frame>_camera.json`, as used by the original loader, and adjacent sequence images accessed through its `get_colors` method. Its full-resolution depth arrays must align with the raw 1024x2048 images before cropping.

KITTI uses the Eigen crop. NYUv2 uses `[45:471, 41:601]`. Cityscapes follows its dedicated evaluator; other dense protocols mask by their valid ranges. nuScenes/Waymo sample predictions at the projected GT pixel coordinates. SUN-RGBD, 7-Scenes, NYUv2 and iBims predictions are capped at 10 m before median scaling, following the intended original generic evaluator. Cityscapes clips after scaling. These implementation details matter when comparing results.

The evaluation protocols cover four training-evaluation and eight zero-shot datasets.
