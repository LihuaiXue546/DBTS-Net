# DBTS-Net


## Overview

DBTS-Net is a dual-branch deep learning network designed for thrombus segmentation in medical imaging. The network integrates vessel attention mechanisms to enhance segmentation accuracy by leveraging vascular structure information.



## Usage

### Installation

```bash
conda create -n dbtsnet python=3.10
conda activate dbtsnet
git clone [<repository-url>](https://github.com/LihuaiXue546/DBTS-Net.git)
cd DBTS-Net
pip install -r requirements.txt
```

Data Preparation

The **FUMPE** dataset  is used in this project.

**Download link:**  
[https://www.kaggle.com/datasets/andrewmvd/pulmonary-embolism-in-ct-images](https://www.kaggle.com/datasets/andrewmvd/pulmonary-embolism-in-ct-images)

**Note:** The dataset contains CTA images from 35 patients with expert-annotated ground truth masks for pulmonary embolism.

The dataset structure should follow this format:
preprocessed_data/
├── csv/
│   ├── train.csv
│   └── val.csv
├── train/
│   ├── image/
│   └── label/
└── val/
    ├── image/
    └── label/
CSV files should contain:
- `image_name`: Filename of the input image
- `mask_name`: Filename of the corresponding label

## Training 

```bash
# Basic training command
python main.py --model_type DBTS-Net

# Training with custom parameters
python main.py --model_type DBTS-Net \
    --batch_size 16 \
    --lr 0.0001 \
    --end_epoch 100 \
    --custom_save_dir /path/to/save/directory \
    --devicenum 0
```

## Inference

```bash
# Run inference on validation set
python infer.py --model_type DBTS-Net \
    --custom_save_dir /path/to/model/checkpoint \
    --result_dir /path/to/save/results \
    --use_best_model
```

