# ChickenAudio Recipe

This recipe provides training scripts for sound classification and interpretation on the ChickenAudio dataset.

**Dataset**: Binary classification (Healthy vs. Unhealthy) using `*__clean__hpss.wav` audio files.

## Classification

Train a CNN14 classifier to distinguish healthy vs. unhealthy chicken sounds:

```bash
cd classification
python train.py hparams/cnn14.yaml --data_folder ../Chicken_Audio_Dataset_Denoised
```

## Interpretation

### NMF (Non-Negative Matrix Factorization)

Train an NMF model with amortized inference:

```bash
cd interpret
python train_nmf.py hparams/nmf.yaml --data_folder ../Chicken_Audio_Dataset_Denoised
```

### L2I (Listen to Interpret)

Train the L2I interpreter (requires a pretrained classifier):

```bash
cd interpret
python train_l2i.py hparams/l2i_cnn14.yaml --data_folder ../Chicken_Audio_Dataset_Denoised
```

### LMAC-TD

Train the LMAC-TD interpreter:

```bash
cd interpret
python train_lmac.py hparams/lmac_cnn14.yaml --data_folder ../Chicken_Audio_Dataset_Denoised
```

## Data Preparation

The data preparation is handled automatically when running training scripts. Manifest files are created in `<data_folder>/manifest/` with a 70/15/15 train/valid/test split.

## Authors

Adapted from the ESC50 recipe by Cem Subakan and Francesco Paissan.
