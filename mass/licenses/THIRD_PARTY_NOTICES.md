# Third-party notices

The repository includes modified or adapted research code. Original copyright notices are retained where available. The license of MaSS contributions does not relicense third-party code.

| Component | Origin | Terms / local notice |
| --- | --- | --- |
| DINOv2 transformer, attention and adapter backbone | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2), with local MaSS modifications | Apache 2.0; [dinoV2/LICENSE](../dinoV2/LICENSE) |
| Depth decoder, shared depth-training utilities and data loaders | [Lite-Mono](https://github.com/noahzn/Lite-Mono), adapted in the original working project | MIT; [LICENSE.lite-mono](LICENSE.lite-mono) |
| Monocular training geometry, pose components, KITTI utilities, base data-loading patterns and Cityscapes loader | [Monodepth2](https://github.com/nianticlabs/monodepth2), including derivatives inherited through the working project | Copyright Niantic 2019. Patent Pending. All rights reserved. Non-commercial terms; [LICENSE.monodepth2](LICENSE.monodepth2) |
| ResNet multi-image input implementation | [torchvision](https://github.com/pytorch/vision) | BSD 3-Clause; [LICENSE.torchvision](LICENSE.torchvision) |

The modified DINOv2 adapter/model files retain the original architecture while removing unused wrappers and imports. The attention module adds CPU fallback to standard attention. MaSS release edits are marked in modified vendored files. The release also specializes training to the requested MaSS branch and relocates dataset protocols into a shared evaluator.

Dataset and pretrained-weight terms are separate from source-code licenses. In particular, inherited Monodepth2 terms include non-commercial restrictions; do not infer unrestricted commercial permission from the root MIT license. Consult the linked upstream terms for any intended use beyond those terms.
