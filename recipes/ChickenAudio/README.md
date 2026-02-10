# ChickenAudio Recipe

This recipe provides training scripts for sound classification and interpretation on poultry/animal respiratory health datasets.

**Supported Datasets**: Binary classification (Healthy vs. Unhealthy) with automatic audio suffix detection (`.wav`, `__clean__hpss.wav`, `_22k.wav`, etc.)

## Quick Start

```bash
# Train classifier (10 seeds)
cd classification
python train_multi_seed.py hparams/cnn14.yaml --data_folder ~/Datasets/YourDataset

# Evaluate AUROC on existing checkpoints
python eval_auroc.py hparams/cnn14.yaml --data_folder ~/Datasets/YourDataset

# Train interpreter
cd ../interpret
python train_l2i.py hparams/l2i_cnn14.yaml --data_folder ~/Datasets/YourDataset
```

---

## Classification

### Single Run Training
```bash
cd classification
python train.py hparams/cnn14.yaml --data_folder ~/Datasets/Chicken_Audio_Dataset_Denoised
```

### Multi-Seed Training (Recommended)
Train 10 classifiers with different random seeds and aggregate AUROC metrics:

```bash
python train_multi_seed.py hparams/cnn14.yaml \
    --data_folder ~/Datasets/Chicken_Audio_Dataset_Denoised \
    --num_seeds 10
```

**Options:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--num_seeds` | Number of seeds to train | 10 |
| `--start_seed` | Resume training from specific seed | 1 |
| `--output_folder` | Custom output folder | `./results/multi_seed/<dataset>` |

**Resume from interrupted training:**
```bash
python train_multi_seed.py hparams/cnn14.yaml \
    --data_folder ~/Datasets/YourDataset \
    --start_seed 5   # Resume from seed 5
```

**Output:** `results/multi_seed/<dataset_name>/auroc_summary.csv`

### AUROC Evaluation (Without Retraining)
Evaluate already-trained checkpoints and compute AUROC summary:

```bash
python eval_auroc.py hparams/cnn14.yaml \
    --data_folder ~/Datasets/Chicken_Audio_Dataset_Denoised
```

---

## Interpretation

### L2I (Listen to Interpret)
> Requires a pretrained classifier (automatically detected from `classification/results/multi_seed/<dataset>/seed_1/`)

```bash
cd interpret
python train_l2i.py hparams/l2i_cnn14.yaml \
    --data_folder ~/Datasets/Chicken_Audio_Dataset_Denoised
```

**Outputs** (saved to `~/Sindhuja_Datasets/L2I_Results/<dataset>/`):
- Listenable explanations: `{sample_id}_listenable.wav`
- Saliency maps: `{sample_id}_saliency.npy` and `{sample_id}_saliency.png`

### LMAC-TD
```bash
cd interpret
python train_lmac.py hparams/lmac_cnn14.yaml \
    --data_folder ~/Datasets/Chicken_Audio_Dataset_Denoised
```

### Multi-Run Interpreter Evaluation
Run evaluation multiple times to get average fidelity/faithfulness with std:

```bash
cd interpret
python eval_multi_run.py hparams/l2i_cnn14.yaml \
    --data_folder ~/Datasets/Chicken_Audio_Dataset_Denoised \
    --num_runs 10
```

**Output:** `results/eval_multi_run_<interpreter>_<dataset>.csv`

---

## Biomarker Extraction

Extract acoustic biomarkers from listenable explanations:

```bash
# Edit biomarker.py to set ROOT and MODEL_NAME, then:
python biomarker.py
```

---

## Data Preparation

Data preparation is **automatic**. Manifests are created in `<data_folder>/manifest/` with 70/15/15 train/valid/test split.

**Audio suffix is auto-detected** - no need to specify for different datasets:
- ChickenAudio: `__clean__hpss.wav`
- SmartEars: `_22k.wav`
- SwineCough: `.wav`

---

## Spectrogram Generation (Optional)

For datasets without pre-computed spectrograms:

```bash
python generate_spectrograms.py --data_folder ~/Datasets/SwineCough
```

---

## Authors

Adapted from the ESC50 recipe by Cem Subakan and Francesco Paissan.
