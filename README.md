<div align="center">

# LeapBot-WA

### World-Anchor Action Models via Predictive Latent Alignments

Pei Liu · Nan Zheng · Lang Zhang · Daojie Peng · Yanan Zhang · Feilong Kong<br>
Mingyue Feng · Jiachao Liu · Yaonong Wang · Qifeng Chen · Jun Ma

[![arXiv](https://img.shields.io/badge/arXiv-2607.23969-b31b1b.svg)](https://arxiv.org/abs/2607.23969)
[![alphaXiv](https://img.shields.io/badge/alphaXiv-Paper%20%26%20Discussion-f97316.svg)](https://www.alphaxiv.org/abs/2607.23969)

**Learning predictive world dynamics for efficient robotic control.**

</div>

Official repository for [**LeapBot-WA: World-Anchor Action Models via Predictive Latent Alignments**](https://arxiv.org/abs/2607.23969).

> **Inference code coming soon.** This initial release contains the project README. The upcoming code release will focus on inference and deployment; training code is outside the planned release scope.

## Overview

LeapBot-WA learns robotic actions through predictive semantic alignment in a latent space. It uses a Joint-Embedding Predictive Architecture (JEPA) as a **World-Anchor**, providing physical dynamics guidance without pixel-level video generation.

The method combines three components:

- **Predictive World-Anchor:** JEPA features provide semantic representations of the environment and its dynamics.
- **Isotropic Semantic Autoencoder (ISAE):** Maps predictive features into a latent space suited to diffusion modeling.
- **Asymmetric Mixture-of-Transformers (MoT):** An Anchor Diffusion Transformer guides an Action Diffusion Transformer during training. The heavy dynamics branch is pruned at inference for efficient action generation.

The [paper](https://arxiv.org/abs/2607.23969) evaluates LeapBot-WA on LIBERO, RoboTwin 2.0, unseen visual conditions, and real-world robotic manipulation.

## Release Status

- [x] Paper available on [arXiv](https://arxiv.org/abs/2607.23969) and [alphaXiv](https://www.alphaxiv.org/abs/2607.23969).
- [x] Project README.
- [ ] Inference code.
- [ ] Installation and model preparation instructions.
- [ ] Inference examples and deployment documentation.

Setup instructions, model asset requirements, and runnable examples will be added with the inference release.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{liu2026leapbotwaworldanchoractionmodels,
  title         = {LeapBot-WA: World-Anchor Action Models via Predictive Latent Alignments},
  author        = {Pei Liu and Nan Zheng and Lang Zhang and Daojie Peng and Yanan Zhang and Feilong Kong and Mingyue Feng and Jiachao Liu and Yaonong Wang and Qifeng Chen and Jun Ma},
  year          = {2026},
  eprint        = {2607.23969},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2607.23969}
}
```
