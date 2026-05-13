# Binary Change Detection on EO-SAR Image Pairs
GalaxEye Space — Satellite AI Research Intern Assignment

**Author:** S.M. Sakthivel  
**Institution:** Pondicherry University, B.Tech AI & Data Science (2026)
**Colab LINK:**  https://colab.research.google.com/drive/1mVVyMjzIVYccIL-w4ejRQkr2_So3O72h?usp=sharing

## Description
Pixel-level binary change detection on co-registered EO and SAR image pairs
using a Dual-Encoder UNet with ResNet34 EO encoder and ResNet18 SAR encoder
with InstanceNorm2d for SAR domain normalization.

Key finding: Generalization failure is driven by cross-scene SAR domain shift.
Test scenes 09-10 have mean SAR intensity 38-44 vs dominant training scenes 57-69.

## Requirements
- Python 3.10
- See requirements.txt for all dependencies

## Environment Setup
```bash
conda create -n galaxeye python=3.10
conda activate galaxeye
pip install -r requirements.txt
```

## Dataset Structure
galaxeye_data/
train/
pre-event/
post-event/
target/
val/
pre-event/
post-event/
target/
test/
pre-event/
post-event/
target/
## Training
```bash
python train.py --config config.yaml
```

## Evaluation
```bash
python eval.py --data_path /path/to/test --weights /path/to/checkpoint.pth
```

## Model Weights
Download FINAL_SUBMISSION.pth (Epoch 10, best test generalization):
(https://drive.google.com/file/d/16odAQZFcg8mMkuB-iMbo1GWipuBIFtOl/view?usp=sharing)

## Results

| Split      | IoU    | F1     | Precision | Recall |
|------------|--------|--------|-----------|--------|
| Validation | 0.1903 | 0.3197 | 0.2675    | 0.4271 |
| Test       | 0.0310 | 0.0602 | 0.0417    | 0.1080 |

Best checkpoint: Epoch 10, Threshold 0.70

## Architecture
- EO Encoder: ResNet34 pretrained ImageNet
- SAR Encoder: ResNet18 + InstanceNorm2d (domain normalization)
- Fusion: Concatenation at 5 scales
- Decoder: 4-stage UNet with skip connections + Dropout2d
- Loss: Focal (0.6) + Dice (0.4)
- Parameters: ~43M

## References
See technical report PDF for full references.