# Official PyTorch Implementation of PhysUSNet: Physics-Guided Deep Learning for Ultrasound Image Segmentation

## Abstract:

Accurate segmentation of ultrasound images plays a crucial role in computer-aided diagnosis by enabling precise tissue characterization. However, segmentation remains challenging due to inherent artifacts such as speckle noise, low contrast, and ambiguous tissue boundaries, which obscure structural details. Furthermore, the high cost of acquiring expert annotations limits the availability of large-scale labeled datasets. While synthetic datasets have been proposed to mitigate data scarcity, they often fail to replicate the full complexity of real-world ultrasound artifacts, reducing their effectiveness in training robust segmentation models. We propose a framework that leverages the physics of wave propagation in tissue to supervise the training of a deep learning model for robust ultrasound segmentation. Our method generates B-mode ultrasound images by solving the acoustic wave equation using a pseudo-spectral time-domain method, which simulates realistic artifacts and captures a wide range of anatomical variations through arbitrary lesion geometries. These synthetic scans serve as pre-training data for our transformer-based models, which are further fine-tuned on clinical ultrasound data through transfer learning to enhance generalization and robustness. Evaluations on public breast ultrasound datasets achieved significant improvements, with Dice scores of 97.96% on BUSI and 90.23% on UDIAT, surpassing the state-of-the-art model. These results highlight the effectiveness of our approach in improving ultrasound segmentation, demonstrating its potential for enhancing clinical diagnostics. These results highlight the effectiveness of our approach in improving ultrasound segmentation, demonstrating its potential for enhancing clinical diagnostics.



## Performance

### Training Performance:
-
### Testing Performance:

-
## Installation

Conda environment (recommended):

```shell
conda env create -f environment.yml


