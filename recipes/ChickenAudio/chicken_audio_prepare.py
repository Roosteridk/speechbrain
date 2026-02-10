"""
Creates data manifest files for the ChickenAudio dataset.

The dataset structure is expected to be:
    data_folder/
        Healthy/
            *__clean__hpss.wav files
        Unhealthy/
            *__clean__hpss.wav files

Authors:
    * Adapted from ESC50 recipe by Cem Subakan and Francesco Paissan
"""

import json
import os
import random

import torch
import torchaudio

import speechbrain as sb
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def detect_audio_suffix(data_folder):
    """
    Automatically detect the audio suffix/pattern from a data folder.
    
    Scans audio files in Healthy/Unhealthy subfolders and determines
    the most common suffix pattern.
    
    Arguments
    ---------
    data_folder : str
        Path to the dataset folder containing Healthy/Unhealthy subdirs.
    
    Returns
    -------
    str
        The detected audio suffix (e.g., ".wav" or "__clean__hpss.wav")
    """
    from collections import Counter
    
    classes = ["Healthy", "Unhealthy"]
    all_files = []
    
    for class_name in classes:
        class_folder = os.path.join(data_folder, class_name)
        if os.path.exists(class_folder):
            for filename in os.listdir(class_folder):
                if filename.endswith(".wav"):
                    all_files.append(filename)
    
    if not all_files:
        logger.warning("No .wav files found for suffix detection, defaulting to '.wav'")
        return ".wav"
    
    # Check for common patterns
    suffix_patterns = []
    for f in all_files:
        if "__clean__hpss.wav" in f:
            suffix_patterns.append("__clean__hpss.wav")
        elif "_22k.wav" in f:
            suffix_patterns.append("_22k.wav")
        else:
            suffix_patterns.append(".wav")
    
    # Return most common pattern
    counter = Counter(suffix_patterns)
    detected_suffix = counter.most_common(1)[0][0]
    logger.info(f"Auto-detected audio_suffix: '{detected_suffix}'")
    return detected_suffix


def prepare_chicken_audio(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    seed=1234,
    skip_manifest_creation=False,
    audio_suffix=None,  # Auto-detect if None
):
    """
    Prepares the json files for the ChickenAudio dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the ChickenAudio dataset is stored.
        Should contain Healthy/ and Unhealthy/ subdirectories.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    train_ratio : float
        Proportion of data to use for training (default 0.7).
    valid_ratio : float
        Proportion of data to use for validation (default 0.15).
    test_ratio : float
        Proportion of data to use for testing (default 0.15).
    seed : int
        Random seed for reproducible splits.
    skip_manifest_creation : bool
        Whether to skip over the manifest creation step.
    audio_suffix : str
        Suffix to filter audio files (default "__clean__hpss.wav").

    Returns
    -------
    None
    """
    if skip_manifest_creation:
        return

    # Auto-detect audio suffix if not specified
    if audio_suffix is None:
        audio_suffix = detect_audio_suffix(data_folder)

    # Validate ratios
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, (
        "Train, valid, and test ratios must sum to 1.0"
    )

    # Set random seed for reproducibility
    random.seed(seed)

    # Collect all audio files
    all_samples = []
    classes = ["Healthy", "Unhealthy"]
    class_to_id = {"Healthy": 0, "Unhealthy": 1}

    for class_name in classes:
        class_folder = os.path.join(data_folder, class_name)
        if not os.path.exists(class_folder):
            logger.warning(f"Class folder not found: {class_folder}")
            continue

        for filename in os.listdir(class_folder):
            # Match all wav files, or filter by specific suffix if not just ".wav"
            if audio_suffix == ".wav":
                should_include = filename.endswith(".wav")
            else:
                should_include = filename.endswith(audio_suffix)
            
            if should_include:
                all_samples.append(
                    {
                        "filename": filename,
                        "class_name": class_name,
                        "class_id": class_to_id[class_name],
                        "class_folder": class_folder,
                    }
                )

    logger.info(f"Found {len(all_samples)} audio files")

    # Stratified split: ensure both classes are in all splits
    samples_by_class = {}
    for sample in all_samples:
        class_name = sample["class_name"]
        if class_name not in samples_by_class:
            samples_by_class[class_name] = []
        samples_by_class[class_name].append(sample)

    train_samples = []
    valid_samples = []
    test_samples = []

    for class_name, class_samples in samples_by_class.items():
        random.shuffle(class_samples)
        n_class = len(class_samples)
        n_train = int(n_class * train_ratio)
        n_valid = int(n_class * valid_ratio)

        train_samples.extend(class_samples[:n_train])
        valid_samples.extend(class_samples[n_train : n_train + n_valid])
        test_samples.extend(class_samples[n_train + n_valid :])

    # Shuffle each split to mix classes
    random.shuffle(train_samples)
    random.shuffle(valid_samples)
    random.shuffle(test_samples)

    logger.info(
        f"Split: train={len(train_samples)}, valid={len(valid_samples)}, test={len(test_samples)}"
    )

    # Create JSON manifests
    create_json(train_samples, data_folder, save_json_train)
    create_json(valid_samples, data_folder, save_json_valid)
    create_json(test_samples, data_folder, save_json_test)


def create_json(samples, data_folder, json_file):
    """
    Creates the json file given a list of samples.

    Arguments
    ---------
    samples : list
        A list of sample dictionaries containing filename, class_name, class_id, class_folder.
    data_folder : str
        Base data folder path.
    json_file : str
        The path of the output json file.
    """
    json_dict = {}

    for sample in samples:
        wav_path = os.path.join(sample["class_folder"], sample["filename"])
        sample_id = os.path.splitext(sample["filename"])[0]

        try:
            # Load audio and get duration
            signal_tensor, sample_rate = torchaudio.load(wav_path)
            signal = signal_tensor.squeeze()
            duration = signal.shape[0] / sample_rate

            # Create entry
            json_dict[sample_id] = {
                "wav": os.path.join(sample["class_name"], sample["filename"]),
                "classID": sample["class_id"],
                "class_string": sample["class_name"],
                "duration": duration,
            }
        except Exception as e:
            logger.warning(f"Error reading {wav_path}: {e}. Skipping.")

    # Create parent directory if needed
    parent_dir = os.path.dirname(json_file)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    # Write JSON
    with open(json_file, mode="w", encoding="utf-8") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created with {len(json_dict)} entries!")


def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""

    data_audio_folder = hparams["audio_data_folder"]
    config_sample_rate = hparams["sample_rate"]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    hparams["resampler"] = torchaudio.transforms.Resample(new_freq=config_sample_rate)

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        wave_file = os.path.join(data_audio_folder, wav)

        sig, read_sr = torchaudio.load(wave_file)

        # If multi-channels, downmix it to a mono channel
        sig = torch.squeeze(sig)
        if len(sig.shape) > 1:
            sig = torch.mean(sig, dim=0)

        # Convert sample rate to required config_sample_rate
        if read_sr != config_sample_rate:
            # Re-initialize sampler if source file sample rate changed
            if read_sr != hparams["resampler"].orig_freq:
                hparams["resampler"] = torchaudio.transforms.Resample(
                    orig_freq=read_sr, new_freq=config_sample_rate
                )
            # Resample audio
            sig = hparams["resampler"].forward(sig)

        # Pad or truncate to fixed length (required for CNN14PSI_stft)
        if "signal_length_s" in hparams:
            target_length = int(hparams["signal_length_s"] * config_sample_rate)
            if sig.shape[0] > target_length:
                # Truncate
                sig = sig[:target_length]
            elif sig.shape[0] < target_length:
                # Pad with zeros
                padding = target_length - sig.shape[0]
                sig = torch.nn.functional.pad(sig, (0, padding))

        sig = sig.float()
        if sig.abs().max() > 0:
            sig = sig / sig.abs().max()
        return sig

    # Define label pipeline
    @sb.utils.data_pipeline.takes("class_string")
    @sb.utils.data_pipeline.provides("class_string", "class_string_encoded")
    def label_pipeline(class_string):
        """The label pipeline."""
        yield class_string
        class_string_encoded = label_encoder.encode_label_torch(class_string)
        yield class_string_encoded

    # Define datasets
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "class_string_encoded"],
        )

    # Load or compute the label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="class_string",
    )

    return datasets, label_encoder
