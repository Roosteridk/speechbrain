#!/usr/bin/env python3
"""Generate spectrograms for animal disease classification datasets.

This script creates mel spectrograms using torchaudio.transforms.MelSpectrogram
to match the format expected by the L2I pipeline.

Usage:
    python generate_spectrograms.py --data_folder ~/Sindhuja_Datasets/SwineCough

Authors:
    * Generated for ChickenAudio L2I pipeline
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB


def compute_spectrogram(
    audio: torch.Tensor,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    target_length: int = 1292,
) -> np.ndarray:
    """Compute dB-normalized mel spectrogram using torchaudio.

    Args:
        audio: Audio signal as torch tensor (1D or 2D)
        sr: Sample rate
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel filterbanks
        target_length: Target time frames (pad/trim to this)

    Returns:
        Spectrogram as float32 array with shape (n_freq, target_length)
    """
    # Ensure audio is 2D (channel, time)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Create MelSpectrogram transform
    mel_transform = MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,  # Power spectrogram
    )

    # Convert to dB scale
    db_transform = AmplitudeToDB(stype="power", top_db=80)

    # Compute mel spectrogram
    mel_spec = mel_transform(audio)

    # Convert to dB
    db_spec = db_transform(mel_spec)

    # Normalize to range [-80, 0]
    db_spec = db_spec - db_spec.max()
    db_spec = torch.clamp(db_spec, min=-80, max=0)

    # Remove channel dimension and convert to numpy
    spec = db_spec.squeeze(0).numpy()

    # Pad or trim to target length
    if spec.shape[1] < target_length:
        # Pad with -80 (silence)
        pad_width = target_length - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=-80)
    elif spec.shape[1] > target_length:
        spec = spec[:, :target_length]

    return spec.astype(np.float32)


def find_audio_files(data_folder: Path) -> list:
    """Find all audio files in Healthy and Unhealthy subdirectories."""
    audio_files = []
    extensions = [".wav", ".mp3", ".flac", ".ogg"]

    for class_name in ["Healthy", "Unhealthy"]:
        class_dir = data_folder / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist, skipping...")
            continue

        for ext in extensions:
            files = list(class_dir.glob(f"*{ext}"))
            for f in files:
                audio_files.append((f, class_name))

    return audio_files


def process_dataset(data_folder: str, overwrite: bool = False):
    """Process all audio files in the dataset and generate spectrograms.

    Args:
        data_folder: Path to dataset root (should contain Healthy/Unhealthy subdirs)
        overwrite: If True, regenerate existing spectrograms
    """
    data_folder = Path(data_folder)

    if not data_folder.exists():
        raise ValueError(f"Data folder does not exist: {data_folder}")

    audio_files = find_audio_files(data_folder)

    if not audio_files:
        raise ValueError(
            f"No audio files found in {data_folder}. "
            "Expected Healthy/ and Unhealthy/ subdirectories with audio files."
        )

    print(f"Found {len(audio_files)} audio files")

    processed = 0
    skipped = 0
    errors = 0

    for audio_path, class_name in audio_files:
        # Create output path
        spec_dir = data_folder / class_name / "spectrograms"
        spec_dir.mkdir(exist_ok=True)

        # Output filename: {original_name}_processed.npy
        stem = audio_path.stem
        out_path = spec_dir / f"{stem}__processed.npy"

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            # Load audio using torchaudio
            audio, sr = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            # Compute spectrogram
            spec = compute_spectrogram(audio.squeeze(0), sr)

            # Save
            np.save(out_path, spec)
            processed += 1

            if processed % 25 == 0:
                print(f"Processed {processed} files...")

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            errors += 1

    print(f"\nComplete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate spectrograms for animal disease classification datasets"
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Path to dataset root containing Healthy/ and Unhealthy/ subdirs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing spectrograms",
    )

    args = parser.parse_args()
    process_dataset(args.data_folder, args.overwrite)


if __name__ == "__main__":
    main()
