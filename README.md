# intrinsic: geometry and topology methods for DL

## Abstract
This repository contains implementation of `intrinsic` package with geometry and topology methods for topological data analysis. It provides tools to compute persistent homology, generate persistence diagrams, calculate total persisentce, persistence entropy, Betti numbers, Euler characteristics, intrinsic dimension (via MLE, MM, PCA, TwoNN), curvature, and magnitude of data representations. 

The detailed description of these properties can be found at [this page](https://drive.google.com/file/d/1XGcXdAt7XSnFjHZR0vK4xVAMW3LySyhj/view?usp=sharing).

## Code Organization
* `functional/`: Contains implementation of `intrinsic` functional.
* `tutorial/`: Contains tutorial on `intrinsic`.
* `examples/`: Contains experiments with `intrinsic`.
* `utils/`: Contains utilitary functions.
* `docs/`: Contains files for Read the Docs documentation.

## Tutorial
* Basic functional applied to 3x3 patches from CIFAR-10 images:  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GIPUQd5Ujtgm0HMp0Q9ufwDw8sInLZNN?usp=sharing)

## Experiments
Geometry and topology properties of
* CIFAR-10 images embeddings on different layers obtained by **VGG-16**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JPPNVgKRCnlRZhZAFgpgWxDNeBHk25AK?usp=sharing)
* MNIST images embeddings on different layers obtained by **SimCLR**:
    * rotation (0,360) transformation: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WQ8OcvCShkSVj3RoZBKlf-KVoScgqcKK?usp=sharing)
    * resize-crop + flip + rotation(0,179) transformations: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FNijhES_qEzJxaKSvdlACRnlvvLNaAE2?usp=sharing)
    * resize-crop + rotation(0,179) transformations: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c2sIAYLRQhMY-j2p-8Vjwi6c9j6sj03s?usp=sharing)





