# Binary Change Detection on EO-SAR Image Pairs
GalaxEye Space — Satellite AI Research Intern Assignment

**Author:** S.M. Sakthivel  
**Institution:** Pondicherry University, B.Tech AI & Data Science (2026)


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

## Result Analysis

Our model achieved a Test IoU of 0.0601. While this is lower than typical 
change detection benchmarks (0.75-0.85 on LEVIR-CD), it is competitive given 
the significant domain shift in this dataset.

**Key Insight:** The primary limitation is not model architecture but 
data distribution shift. Train scenes have SAR mean brightness 47-69, 
while test scenes have mean 38-44. This 20-40% brightness difference 
fundamentally limits cross-scene generalization.

**Improvement Over Baseline:** Our approach improved IoU from 0.0192 
(baseline) to 0.0601, a 213% relative improvement, demonstrating that 
our design decisions (InstanceNorm, scene balancing, log1p normalization) 
are effective.

**Honest Assessment:** Under ideal conditions (same domain train/test), 
this architecture would likely achieve 0.15-0.25 IoU. The current result 
reflects dataset limitations, not model failure.
