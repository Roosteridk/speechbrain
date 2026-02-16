import os

import glob

import numpy as np

import librosa


ROOT = "/home/arda/Sindhuja_Datasets/LMAC-TD/Chicken_Audio_Dataset_Denoised"  # listenable explanations root directory. filename should be of form: <any prefix here>_listenable.wav

MODEL_NAME = "lmac-td"  # coughlime, l2i, etc

HEALTHY_DIR = os.path.join(ROOT, "Healthy")

UNHEALTHY_DIR = os.path.join(ROOT, "Unhealthy")


OUT_CSV = os.path.join(ROOT, f"bio_markers_listenable_original_data_{MODEL_NAME}.csv")


# feature params

SR_TARGET = None  #

N_FFT = 1024

HOP = 512

EPS = 1e-12


# -----------------------------

# Helpers

# -----------------------------


def shannon_entropy_from_psd(psd: np.ndarray) -> float:
    psd = psd.astype(np.float64)

    s = psd.sum()

    if s <= 0:
        return 0.0

    p = psd / (s + EPS)

    return float(-(p * np.log(p + EPS)).sum())


def peak_frequency_from_psd(freqs: np.ndarray, psd: np.ndarray) -> float:
    if psd.size == 0:
        return 0.0

    return float(freqs[int(np.argmax(psd))])


def spectral_centroid_mean(y: np.ndarray, sr: int) -> float:
    c = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP)

    return float(np.mean(c))


def rms_amplitude(y: np.ndarray) -> float:
    # energy-based amplitude

    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]

    return float(np.mean(rms))


def hnr_harmonic_to_noise(y: np.ndarray, sr: int) -> float:
    if np.allclose(y, 0):
        return 0.0

    y_harm, y_perc = librosa.effects.hpss(y)

    # treat "noise" as what isn't harmonic

    y_noise = y - y_harm

    Eh = float(np.sum(y_harm**2))

    En = float(np.sum(y_noise**2))

    if En <= 0:
        return float("inf")

    return float(10.0 * np.log10((Eh + EPS) / (En + EPS)))


def spectral_entropy(y: np.ndarray, sr: int) -> float:
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2  # power

    psd = np.mean(S, axis=1)  # average over time -> [F]

    return shannon_entropy_from_psd(psd)


def mean_power_spectrum(y: np.ndarray, sr: int):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2

    psd = np.mean(S, axis=1)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    return freqs, psd


def parse_id_from_filename(fn: str) -> str:
    # e.g., "1__clean____processed__ig_listenable.wav" -> "1"

    base = os.path.basename(fn)

    return base.split("__")[0]


def iter_files():
    files = []

    files += [
        (p, "Healthy")
        for p in sorted(glob.glob(os.path.join(HEALTHY_DIR, "*_listenable.wav")))
    ]

    files += [
        (p, "Unhealthy")
        for p in sorted(glob.glob(os.path.join(UNHEALTHY_DIR, "*_listenable.wav")))
    ]

    return files


def write_csv(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    header = [
        "status",
        "patient_id",
        "path",
        "entropy",
        "peakfreq_hz",
        "hnr_db",
        "spectral_centroid_hz",
        "rms",
    ]

    with open(out_csv, "w") as f:
        f.write(",".join(header) + "\n")

        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


def summarize(rows, status):
    # columns: entropy(3), peakfreq(4), hnr(5), centroid(6), rms(7)

    arr = np.array(
        [
            [float(r[3]), float(r[4]), float(r[5]), float(r[6]), float(r[7])]
            for r in rows
        ],
        dtype=float,
    )

    if arr.size == 0:
        print(f"[{status}] no samples")

        return

    means = arr.mean(axis=0)

    stds = arr.std(axis=0)

    names = ["entropy", "peakfreq_hz", "hnr_db", "spectral_centroid_hz", "rms"]

    print(f"\n[{status}] n={arr.shape[0]}")

    for i, nm in enumerate(names):
        print(f"  {nm}: mean={means[i]:.4f} | std={stds[i]:.4f}")


def main():
    items = iter_files()

    if len(items) == 0:
        raise RuntimeError(
            "No listenable wavs found.\n"
            f"Tried:\n  {HEALTHY_DIR}\n  {UNHEALTHY_DIR}\n"
            "Expected filenames like: *_listenable.wav"
        )

    rows = []

    for idx, (wav_path, status) in enumerate(items):
        y, sr = librosa.load(wav_path, sr=SR_TARGET, mono=True)

        if y.size == 0:
            continue

        # markers

        ent = spectral_entropy(y, sr)

        freqs, psd = mean_power_spectrum(y, sr)

        peakf = peak_frequency_from_psd(freqs, psd)

        hnr = hnr_harmonic_to_noise(y, sr)

        centroid = spectral_centroid_mean(y, sr)

        amp = rms_amplitude(y)

        pid = parse_id_from_filename(wav_path)

        rows.append([status, pid, wav_path, ent, peakf, hnr, centroid, amp])

        if (idx + 1) % 25 == 0:
            print(f"processed {idx + 1}/{len(items)}")

    write_csv(rows, OUT_CSV)

    print("\nSaved:", OUT_CSV)

    healthy_rows = [r for r in rows if r[0] == "Healthy"]

    unhealthy_rows = [r for r in rows if r[0] == "Unhealthy"]

    summarize(healthy_rows, "Healthy")

    summarize(unhealthy_rows, "Unhealthy")


if __name__ == "__main__":
    main()
