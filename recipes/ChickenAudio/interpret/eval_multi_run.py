#!/usr/bin/env python3
"""Run evaluation metrics multiple times and report average fidelity/faithfulness.

This script evaluates a trained LMAC or L2I interpreter 10 times and computes
average input_fidelity and faithfulness_mean with standard deviations.

Usage:
    # For LMAC
    python eval_multi_run.py hparams/lmac_cnn14.yaml \
        --data_folder ~/Datasets/Chicken_Audio_Dataset_Denoised \
        --num_runs 10

    # For L2I
    python eval_multi_run.py hparams/l2i_cnn14.yaml \
        --data_folder ~/Datasets/Chicken_Audio_Dataset_Denoised \
        --num_runs 10

Authors:
    * Generated for ChickenAudio evaluation pipeline
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chicken_audio_prepare import dataio_prep, prepare_chicken_audio

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main


def detect_explainer_type(hparams_file: str) -> str:
    """Detect whether hparams file is for LMAC or L2I."""
    basename = os.path.basename(hparams_file).lower()
    if "lmac" in basename:
        return "lmac"
    elif "l2i" in basename:
        return "l2i"
    else:
        raise ValueError(f"Cannot detect explainer type from {hparams_file}")


def run_single_evaluation(
    hparams_file: str,
    data_folder: str,
    run_idx: int,
    base_overrides: str = "",
) -> dict:
    """Run a single evaluation and return metrics."""
    
    seed = 1000 + run_idx  # Different seed per run
    
    # Build overrides
    overrides = base_overrides
    overrides += f"\nseed: {seed}"
    overrides += f"\ndata_folder: {data_folder}"
    overrides += "\ntest_only: True"
    overrides += "\nsave_interpretations: False"  # Skip saving for speed
    
    # Infer dataset_name
    norm_path = os.path.normpath(data_folder)
    dataset_name = os.path.basename(norm_path)
    overrides += f"\ndataset_name: {dataset_name}"
    
    # Infer pretrained_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(
        script_dir, "..", "classification", "results",
        "multi_seed", dataset_name, "seed_1", "save"
    )
    save_dir = os.path.normpath(save_dir)
    
    if os.path.isdir(save_dir):
        ckpt_folders = sorted([
            d for d in os.listdir(save_dir)
            if d.startswith("CKPT") and os.path.isdir(os.path.join(save_dir, d))
        ])
        if ckpt_folders:
            pretrained_path = os.path.join(save_dir, ckpt_folders[-1])
            overrides += f"\npretrained_path: {pretrained_path}"
    
    # Set seed
    sb.utils.seed_everything(seed)
    
    # Load hyperparameters
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    # Prepare dataset
    run_on_main(
        prepare_chicken_audio,
        kwargs={
            "data_folder": data_folder,
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "skip_manifest_creation": hparams.get("skip_manifest_creation", False),
        },
    )
    
    datasets, label_encoder = dataio_prep(hparams)
    hparams["label_encoder"] = label_encoder
    
    # Freeze classifier
    hparams["embedding_model"].eval()
    hparams["classifier"].eval()
    hparams["embedding_model"].requires_grad_(False)
    hparams["classifier"].requires_grad_(False)
    
    # Detect explainer type and create brain
    explainer_type = detect_explainer_type(hparams_file)
    
    if explainer_type == "lmac":
        from train_lmac import LMAC
        brain_class = LMAC
    else:
        from train_l2i import L2I
        brain_class = L2I
        hparams["nmf_decoder"].eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    brain = brain_class(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts={"device": device},
        checkpointer=hparams["checkpointer"],
    )
    
    # Load pretrained
    if "pretrained" in hparams and hparams.get("use_pretrained", True):
        run_on_main(hparams["pretrained"].collect_files)
        hparams["pretrained"].load_collected()
    
    # Move models to device
    hparams["embedding_model"].to(device)
    hparams["classifier"].to(device)
    if explainer_type == "l2i":
        hparams["nmf_decoder"].to(device)
    
    # Recover checkpoint
    brain.checkpointer.recover_if_possible(min_key="loss")
    
    # Run evaluation
    brain.evaluate(
        test_set=datasets["test"],
        min_key="loss",
        progressbar=False,
        test_loader_kwargs=hparams["dataloader_options"],
    )
    
    # Extract metrics
    fidelity = torch.Tensor(brain.inp_fid.scores).mean().item()
    faithfulness = torch.Tensor(brain.faithfulness.scores).mean().item()
    ai = torch.Tensor(brain.AI.scores).mean().item()
    ad = torch.Tensor(brain.AD.scores).mean().item()
    ag = torch.Tensor(brain.AG.scores).mean().item()
    sps = torch.Tensor(brain.sps.scores).mean().item()
    comp_tensor = torch.Tensor(brain.comp.scores)
    comp_tensor = comp_tensor[~torch.isnan(comp_tensor)]
    comp = comp_tensor.mean().item() if len(comp_tensor) > 0 else float("nan")
    
    return {
        "run": run_idx,
        "seed": seed,
        "input_fidelity": fidelity,
        "faithfulness_mean": faithfulness,
        "AI": ai,
        "AD": ad,
        "AG": ag,
        "SPS": sps,
        "COMP": comp,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation multiple times and report average metrics"
    )
    parser.add_argument("hparams_file", type=str, help="Path to hyperparameter YAML file")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of evaluation runs")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV file path")
    
    args = parser.parse_args()
    
    explainer_type = detect_explainer_type(args.hparams_file)
    dataset_name = os.path.basename(os.path.normpath(args.data_folder))
    
    print(f"\n{'='*60}")
    print(f"Multi-Run Evaluation: {explainer_type.upper()}")
    print(f"Dataset: {dataset_name}")
    print(f"Number of runs: {args.num_runs}")
    print(f"{'='*60}\n")
    
    results = []
    for run_idx in range(1, args.num_runs + 1):
        print(f"\n--- Run {run_idx}/{args.num_runs} ---")
        try:
            result = run_single_evaluation(
                args.hparams_file,
                args.data_folder,
                run_idx,
            )
            results.append(result)
            print(f"  Fidelity: {result['input_fidelity']:.4f}")
            print(f"  Faithfulness: {result['faithfulness_mean']:.4f}")
            print(f"  AI: {result['AI']:.4f}  AD: {result['AD']:.4f}  AG: {result['AG']:.4f}")
            print(f"  SPS: {result['SPS']:.4f}  COMP: {result['COMP']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "run": run_idx,
                "seed": 1000 + run_idx,
                "input_fidelity": float("nan"),
                "faithfulness_mean": float("nan"),
                "AI": float("nan"),
                "AD": float("nan"),
                "AG": float("nan"),
                "SPS": float("nan"),
                "COMP": float("nan"),
            })
    
    # Compute summary statistics
    metric_keys = ["input_fidelity", "faithfulness_mean", "AI", "AD", "AG", "SPS", "COMP"]
    metric_arrays = {}
    for key in metric_keys:
        vals = np.array([r[key] for r in results if not np.isnan(r[key])])
        metric_arrays[key] = vals
    
    print(f"\n{'='*60}")
    print(f"SUMMARY - {explainer_type.upper()} on {dataset_name}")
    print(f"{'='*60}")
    print(f"Successful runs: {len(metric_arrays['input_fidelity'])}/{args.num_runs}")
    
    for key in metric_keys:
        vals = metric_arrays[key]
        if len(vals) > 0:
            print(f"\n{key:>20s}: {vals.mean():.4f} ± {vals.std():.4f}")
            print(f"{'':>20s}  Range: [{vals.min():.4f}, {vals.max():.4f}]")
    
    # Save CSV
    csv_fieldnames = ["run", "seed"] + metric_keys
    output_csv = args.output_csv or f"./results/eval_multi_run_{explainer_type}_{dataset_name}.csv"
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
        # Write summary row
        f.write(f"\n# Summary\n")
        for key in metric_keys:
            vals = metric_arrays[key]
            if len(vals) > 0:
                f.write(f"# {key}: {vals.mean():.4f} ± {vals.std():.4f}\n")
    
    print(f"\nResults saved to: {output_csv}")


if __name__ == "__main__":
    main()
