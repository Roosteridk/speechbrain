#!/usr/bin/env python3
"""Train multiple classifiers with different seeds and aggregate AUROC results.

This script trains 10 classifiers with different random seeds on the same dataset
and outputs a summary CSV with AUROC metrics for each run.

Usage:
    python train_multi_seed.py hparams/cnn14.yaml --data_folder ~/Sindhuja_Datasets/Chicken_Audio_Dataset_Denoised

Authors:
    * Generated for ChickenAudio L2I pipeline
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from sklearn.metrics import roc_auc_score

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chicken_audio_prepare import dataio_prep, prepare_chicken_audio

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.train_logger import FileTrainLogger


class ChickenAudioBrainWithAUROC(sb.core.Brain):
    """Class for classifier training with AUROC tracking."""

    def compute_forward(self, batch, stage):
        """Computation pipeline based on an encoder + sound classifier."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        if hasattr(self.hparams, "augmentation") and stage == sb.Stage.TRAIN:
            wavs, lens = self.hparams.augmentation(wavs, lens)

        X_stft = self.modules.compute_stft(wavs)
        net_input = sb.processing.features.spectral_magnitude(
            X_stft, power=self.hparams.spec_mag_power
        )
        if hasattr(self.hparams, "use_melspectra") and self.hparams.use_melspectra:
            net_input = self.modules.compute_fbank(net_input)

        if (not self.hparams.use_melspectra) or self.hparams.use_log1p_mel:
            net_input = torch.log1p(net_input)

        embeddings = self.modules.embedding_model(net_input)
        if isinstance(embeddings, tuple):
            embeddings, _ = embeddings

        if embeddings.ndim == 4:
            embeddings = embeddings.mean((-1, -2))

        outputs = self.modules.classifier(embeddings)
        if outputs.ndim == 2:
            outputs = outputs.unsqueeze(1)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using class-id as label."""
        predictions, lens = predictions
        classid, _ = batch.class_string_encoded

        N_augments = int(predictions.shape[0] / classid.shape[0])
        classid = torch.cat(N_augments * [classid], dim=0)

        target = F.one_hot(classid.squeeze(), num_classes=self.hparams.out_n_neurons)
        loss = -(F.log_softmax(predictions.squeeze(1), 1) * target).sum(1).mean()

        if stage == sb.Stage.TEST:
            probs = F.softmax(predictions.squeeze(1), dim=1)
            self.test_preds.append(probs.cpu().detach())
            self.test_labels.append(classid.cpu().detach())

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch."""
        if stage == sb.Stage.TEST:
            self.test_preds = []
            self.test_labels = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        if stage == sb.Stage.VALID:
             self.checkpointer.save_and_keep_only(
                    meta={"error": stage_loss}, min_keys=["error"],
             )

        if stage == sb.Stage.TEST and self.test_preds:
            all_preds = torch.cat(self.test_preds, dim=0).numpy()
            all_labels = torch.cat(self.test_labels, dim=0).numpy().squeeze()

            try:
                # Check for NaN in predictions
                if np.isnan(all_preds).any():
                    print(f"WARNING: NaN values found in predictions!")
                    self.auroc = float("nan")
                    return

                # Check label distribution
                unique_labels = np.unique(all_labels)
                print(f"Test set: {len(all_labels)} samples, labels: {unique_labels}, "
                      f"distribution: {np.bincount(all_labels.astype(int))}")

                if len(unique_labels) < 2:
                    print(f"ERROR: Only one class in test set, cannot compute AUROC")
                    self.auroc = float("nan")
                    return

                if self.hparams.out_n_neurons == 2:
                    self.auroc = roc_auc_score(all_labels, all_preds[:, 1])
                else:
                    self.auroc = roc_auc_score(
                        all_labels, all_preds, multi_class="ovr", average="macro"
                    )
            except Exception as e:
                print(f"AUROC calculation failed: {e}")
                self.auroc = float("nan")


def train_single_seed(hparams_file: str, data_folder: str, seed: int, base_output: str) -> dict:
    """Train a single classifier with given seed and return AUROC."""
    output_folder = os.path.join(base_output, f"seed_{seed}")

    # Load hyperparameters with overrides
    overrides = {
        "seed": seed,
        "data_folder": data_folder,
        "output_folder": output_folder,
    }

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Set seed
    sb.utils.seed_everything(seed)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=output_folder,
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Setup logger
    hparams["train_logger"] = FileTrainLogger(
        save_file=os.path.join(output_folder, "train_log.txt")
    )

    # Prepare dataset
    run_on_main(
        prepare_chicken_audio,
        kwargs={
            "data_folder": data_folder,
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "skip_manifest_creation": hparams.get("skip_manifest_creation", False),
            "audio_suffix": hparams.get("audio_suffix", "__clean__hpss.wav"),
        },
    )

    datasets, label_encoder = dataio_prep(hparams)
    hparams["label_encoder"] = label_encoder

    # Create brain
    brain = ChickenAudioBrainWithAUROC(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        checkpointer=hparams["checkpointer"],
    )

    # Train
    brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Evaluate
    brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )

    auroc = getattr(brain, "auroc", None)

    return {
        "seed": seed,
        "auroc": auroc,
        "output_folder": output_folder,
        "success": auroc is not None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train multiple classifiers with different seeds"
    )
    parser.add_argument("hparams_file", type=str, help="Path to hyperparameter YAML file")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of seeds (default: 10)")
    parser.add_argument("--start_seed", type=int, default=1, help="Seed to start from (for resuming)")
    parser.add_argument("--output_folder", type=str, default=None, help="Base output folder")

    args = parser.parse_args()

    dataset_name = Path(args.data_folder).name
    base_output = args.output_folder or f"./results/multi_seed/{dataset_name}"
    os.makedirs(base_output, exist_ok=True)

    results = []
    for seed in range(args.start_seed, args.num_seeds + 1):
        print(f"\n{'='*60}")
        print(f"Training with seed {seed}/{args.num_seeds}")
        print(f"{'='*60}\n")

        result = train_single_seed(args.hparams_file, args.data_folder, seed, base_output)
        results.append(result)

        if result["auroc"]:
            print(f"Seed {seed} AUROC: {result['auroc']:.4f}")

    # Save results
    csv_file = os.path.join(base_output, "auroc_summary.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "auroc", "success", "output_folder"])
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    aurocs = np.array([r["auroc"] for r in results if r["auroc"] is not None])
    print(f"\n{'='*60}")
    print(f"SUMMARY - {dataset_name}")
    print(f"{'='*60}")
    print(f"Successful runs: {len(aurocs)}/{len(results)}")
    if len(aurocs) > 0:
        print(f"AUROC: {aurocs.mean():.4f} ± {aurocs.std():.4f}")
        print(f"  Min: {aurocs.min():.4f}, Max: {aurocs.max():.4f}")
    print(f"\nResults saved to: {csv_file}")


if __name__ == "__main__":
    main()
