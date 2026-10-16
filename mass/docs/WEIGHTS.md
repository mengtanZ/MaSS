# Weights

## Training initialization

Download the official **DINOv2 ViT-B/14 with registers** checkpoint:

- [Official DINOv2 repository and pretrained-model table](https://github.com/facebookresearch/dinov2)
- [ViT-B/14 register checkpoint](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth)

Save it as `mass/weights/dinov2_vitb14_reg4_pretrain.pth`, or pass `--dino_weights /path/to/file.pth` to training. The loader preserves the original architecture and pretrained state-key names. Only spatial-prior, injector, extractor, level embeddings, projection/fusion modules, depth decoder and pose network parameters are trained; the DINOv2 transformer parameters remain frozen.

The pose encoder uses the torchvision ImageNet-pretrained ResNet-18 checkpoint, downloaded to PyTorch's cache on first use. For offline use, populate that cache in advance.

The historical Lite-Mono `--mypretrain` flag had no matching keys in the active MaSS encoder; it is a compatibility option, not another initialization requirement. The current model uses a convolutional spatial prior, not the unused experimental Lite-Mono wrapper.

## Evaluation

A complete MaSS evaluation folder contains `encoder.pth` and `depth.pth`. The encoder file includes `height`, `width`, `use_stereo` metadata and the frozen DINOv2 parameters. The evaluator loads all model parameters strictly and constructs the architecture without downloading separate pretrained weights.

The mixed-dataset checkpoint is hosted separately on [Hugging Face](https://huggingface.co/mengtanZ/MaSS/tree/main/mass_mix). Download both files into the default evaluation folder:

```bash
mkdir -p mass/weights/mass_mix
curl -L --fail https://huggingface.co/mengtanZ/MaSS/resolve/main/mass_mix/encoder.pth -o mass/weights/mass_mix/encoder.pth
curl -L --fail https://huggingface.co/mengtanZ/MaSS/resolve/main/mass_mix/depth.pth -o mass/weights/mass_mix/depth.pth
bash scripts/testing_submit.sh
```

The checkpoint uses a 384×640 input resolution and was trained jointly on DDAD, NYUv2, KITTI and nuScenes. Downloaded weights remain excluded from Git. Single-dataset checkpoints are not currently linked here; train those models using the corresponding commented recipes in `scripts/training_submit.sh`.
