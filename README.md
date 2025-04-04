# PhyUSFormer: Physics-Guided Ultrasound Segmentation Transformer

![PhyUSFormer Banner](https://img.shields.io/badge/PhyUSFormer-Ultrasound%20Segmentation-blue)

## Abstract

The accurate segmentation of ultrasound images plays a crucial role in computer-aided diagnosis by enabling precise tissue characterization. However, segmentation remains challenging due to inherent artifacts such as speckle noise, low contrast, and ambiguous tissue boundaries, which obscure structural details. Furthermore, the high cost of acquiring expert annotations limits the availability of large-scale labeled datasets. 

While synthetic datasets have been proposed to mitigate data scarcity, they often fail to replicate the full complexity of real-world ultrasound artifacts, reducing their effectiveness in training robust segmentation models. 

We propose a framework that leverages the physics of wave propagation in tissue to supervise the training of a deep learning model for robust ultrasound segmentation. Our method generates B-mode ultrasound images by solving the acoustic wave equation using a pseudo-spectral time-domain method, which simulates realistic artifacts and captures a wide range of anatomical variations through arbitrary lesion geometries. These synthetic scans serve as pre-training data for our transformer-based models, which are further fine-tuned on clinical ultrasound data through transfer learning to enhance generalization and robustness. 

Evaluations on public breast ultrasound datasets achieved significant improvements, with Dice scores of 90.57% on BUSI and 89.82% on UDIAT, surpassing the state-of-the-art model. These results highlight the effectiveness of our approach in improving ultrasound segmentation, demonstrating its potential for enhancing clinical diagnostics.

## Highlights

- Flexible pipeline to generate large, high-quality segmentation datasets
- Outperforms previous state-of-the-art ("Beyond SAM") on BUSI and UDIAT datasets
- Physics-guided approach that accurately simulates ultrasound artifacts
- Transformer-based architecture optimized for medical image segmentation

## Installation

```bash
# Clone the repository
git clone https://github.com/ghafoorharis/PhyUSFormer.git
cd PhyUSFormer

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (see Models section)
```

## Usage

### Training

```bash
# Train the model with default settings
python train.py

# Train with specific fold ID
python train.py --fold_id 4

# Train with custom log and save directories
python train.py --fold_id 0 --log_dir "runs/experiment_1" --save_dir "weights/experiment_1"

# Train with Weights & Biases logging
python train.py --use_wandb
```

### Evaluation

For detailed evaluation, please refer to the `eval.ipynb` notebook which contains step-by-step instructions for evaluating models on the benchmark datasets.

## Results

### Quantitative Results

**Comparison of Base Models on Synthetic Datasets**

| Model | Dataset | Dice Score | IoU | Precision | Recall |
|-------|---------|------------|-----|-----------|--------|
| UNet  | Quadratic Bezier | 0.89 | 0.80 | 0.91 | 0.87 |
| SegFormer | Quadratic Bezier | 0.92 | 0.85 | 0.93 | 0.91 |
| PhyUSFormer | Quadratic Bezier | 0.94 | 0.88 | 0.95 | 0.93 |
| UNet  | Cubic Bezier | 0.87 | 0.78 | 0.88 | 0.86 |
| SegFormer | Cubic Bezier | 0.90 | 0.83 | 0.91 | 0.89 |
| PhyUSFormer | Cubic Bezier | 0.93 | 0.87 | 0.94 | 0.92 |

**Comparison of PhyUSFormer with SOTA on Benchmark Datasets**

| Model | BUSI Dataset | UDIAT Dataset |
|-------|--------------|---------------|
| U-Net | 85.71% | 84.25% |
| Attention U-Net | 86.32% | 85.18% |
| SwinUNet | 87.45% | 86.91% |
| TransUNet | 88.12% | 87.33% |
| Beyond SAM | 89.40% | 88.75% |
| PhyUSFormer (Ours) | **90.57%** | **89.82%** |

### Qualitative Results

Our model demonstrates superior boundary delineation and noise robustness compared to existing approaches, particularly in challenging cases with low contrast and complex lesion morphology.

## Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| Quadratic Bezier Curve | Synthetic dataset generated using quadratic Bezier curves to model lesion boundaries | [Download](https://drive.google.com/drive/folders/19xJpTnRzEUu9yBBdiZBxpWI5ljccu5bb?usp=sharing) |
| Cubic Bezier Curve | Synthetic dataset with more complex lesion boundaries using cubic Bezier curves | [Download](https://drive.google.com/drive/folders/19xJpTnRzEUu9yBBdiZBxpWI5ljccu5bb?usp=sharing) |
| Real World Phantoms Based Scans | Scans based on realistic tissue-mimicking phantoms | [Download](https://drive.google.com/drive/folders/19xJpTnRzEUu9yBBdiZBxpWI5ljccu5bb?usp=sharing) |
| Benchmark Datasets (BUSI and UDIAT) | Public breast ultrasound benchmark datasets | [Download](https://drive.google.com/drive/folders/19xJpTnRzEUu9yBBdiZBxpWI5ljccu5bb?usp=sharing) |

## Pre-trained Models

Please download the pre-trained weights from the following drive link and place them in the "weights" folder:

[Download Pre-trained Models](https://drive.google.com/drive/folders/1D-YaHevYGR69wwR0UMNLFRCibDtuWT7H?usp=sharing)

## Conference Proceedings
This work is currently in the process of submission.

<!---
This work has been accepted in the proceedings of the International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI 2023). The full paper will be available in the conference proceedings.
-->

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- We thank the contributors of the BUSI and UDIAT datasets for making their data publicly available
- Special thanks to the open-source community for their valuable tools and libraries

## Contact

For questions or issues, please open an issue on GitHub or contact the authors directly.

GitHub: [https://github.com/ghafoorharis/PhyUSFormer.git](https://github.com/ghafoorharis/PhyUSFormer.git)


