# Modified InterFaceGAN for PGGAN

This repo is modified form [interfacegan](https://github.com/genforce/interfacegan)

[[Paper (CVPR)](https://arxiv.org/pdf/1907.10786.pdf)]
[[Paper (TPAMI)](https://arxiv.org/pdf/2005.09635.pdf)]
[[Project Page](https://genforce.github.io/interfacegan/)]
[[Demo](https://www.youtube.com/watch?v=uoftpl3Bj6w)]
[[Colab](https://colab.research.google.com/github/genforce/interfacegan/blob/master/docs/InterFaceGAN.ipynb)]

## How to Use

Pick up a model, pick up a boundary, pick up a latent code, and then EDIT!

```bash
# Before running the following code, please first download
# the pre-trained ProgressiveGAN model on CelebA-HQ dataset,
# and then place it under the folder ".models/pretrain/".
python edit.py -a heavy_makeup -o results --steps 3
```

### Tools
- `train_classifier.py`: This script will train a ResNet18 binary classifier for different attribute in CelebAHQ dataset.

- `generate_data.py`: This script can be used for data preparation. It will generate a collection of syntheses (images are saved for further attribute prediction) as well as save the input latent codes.

- `classify_samples.py`: This script will create `scores.npy`, classifying generated image into certain params.

- `train_boundary.py`: This script can be used for boundary searching.

- `edit.py`: This script can be usd for semantic face editing.

## Usage

## Train classifier
```bash
python train_classifier.py -a heavy_makeup -i ..\data\celeba\img_align_celeba\ 
```
After training, rename the best `pth` removing the epoch and accuracy info. e.g `resnet18_heavy_makeup_e_19_acc_0.9892.pth` into `resnet18_heavy_makeup.pth`

### Prepare data

```bash
python generate_data.py -o data -n 10000
```

### Predict Attribute Score

```bash
python classify_samples.py -a heavy_makeup -i data 
```

### Search Semantic Boundary

```bash
python train_boundary.py -a heavy_makeup -i data
```

### Compute Conditional Boundary (Optional)

This step is optional. It depends on whether conditional manipulation is needed. Users can use function `project_boundary()` in file `utils/manipulator.py` to compute the projected direction.

## Boundaries Description

We provided following boundaries in folder `boundaries/`. The boundaries can be more accurate if stronger attribute predictor is used.

- ProgressiveGAN model trained on CelebA-HQ dataset:
  - Single boundary:
    - `pggan_celebahq_pose_boundary.npy`: Pose.
    - `pggan_celebahq_smile_boundary.npy`: Smile (expression).
    - `pggan_celebahq_age_boundary.npy`: Age.
    - `pggan_celebahq_gender_boundary.npy`: Gender.
    - `pggan_celebahq_eyeglasses_boundary.npy`: Eyeglasses.
    - `pggan_celebahq_quality_boundary.npy`: Image quality.
  - Conditional boundary:
    - `pggan_celebahq_age_c_gender_boundary.npy`: Age (conditioned on gender).
    - `pggan_celebahq_age_c_eyeglasses_boundary.npy`: Age (conditioned on eyeglasses).
    - `pggan_celebahq_age_c_gender_eyeglasses_boundary.npy`: Age (conditioned on gender and eyeglasses).
    - `pggan_celebahq_gender_c_age_boundary.npy`: Gender (conditioned on age).
    - `pggan_celebahq_gender_c_eyeglasses_boundary.npy`: Gender (conditioned on eyeglasses).
    - `pggan_celebahq_gender_c_age_eyeglasses_boundary.npy`: Gender (conditioned on age and eyeglasses).
    - `pggan_celebahq_eyeglasses_c_age_boundary.npy`: Eyeglasses (conditioned on age).
    - `pggan_celebahq_eyeglasses_c_gender_boundary.npy`: Eyeglasses (conditioned on gender).
    - `pggan_celebahq_eyeglasses_c_age_gender_boundary.npy`: Eyeglasses (conditioned on age and gender).
