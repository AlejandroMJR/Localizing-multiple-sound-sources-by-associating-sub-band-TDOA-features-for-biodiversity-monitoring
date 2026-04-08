import numpy as np
import os
import librosa
import soundfile as sf
from utils import clear_folder
import math
from ForestIR.code.ForestReverb import generateSampleForestIR, generateMicSignals
"""
ForestIR taken from Kaneko, S., & Gamper, H. (2021, October). A fast forest reverberator using single scattering cylinders. In 2021 IEEE 23rd International Workshop on Multimedia Signal Processing (MMSP) (pp. 1-5). IEEE.
"""

def _simulate(
    nMics=4,
    nSources=5,
    environmentSize=30,
    noisePower=50,
    sourcePower=80,
    saveSignals=True,
    fs=48000,
    max_duration=3,
    sources_folder="audios/cleanSources",
):
    """
    Simulates audio signals captured by sparse microphones from multiple sources
    in a (ForestIR) environment.

    Args:
        nMics (int): Number of microphones (sparse sensors).
        nSources (int): Number of active sources.
        environmentSize (float): Square environment side length (meters).
        noisePower (float): Noise power in dB SPL.
        sourcePower (float): Source power in dB SPL.
        saveSignals (bool): Whether to save mic signals and pooled sources to disk.
        fs (int): Sampling rate.
        max_duration (float): Truncate/pad sources to this duration in seconds.
        sources_folder (str): Folder containing source*.wav.

    Returns:
        fs (int): Sampling frequency.
        micPos (np.ndarray): Microphone positions [nMics, 2].
        signals (np.ndarray): Simulated microphone signals w/ noise [nMics, nSamples].
        srcPos (np.ndarray): Source positions [nSources, 2].
    """

    # Positions
    micPos, srcPos = simulation_setup_sparse_mics(nMics, nSources, environmentSize)

    # Load and standardize sources
    max_samples = int(max_duration * fs)
    allSources = load_sources_from_folder(
        sources_folder=sources_folder,
        fs=fs,
        max_samples=max_samples,
        pad_to_max=True,
        prefix="source",
        ext=".wav"
    )

    if nSources > len(allSources):
        raise ValueError(
            f"Requested nSources ({nSources}) exceeds available source files ({len(allSources)})."
        )

    if saveSignals:
        clear_folder("audios/micSignals")
        clear_folder("audios/pooledSources")

    # Randomly sample nSources
    selected_idx = np.random.choice(len(allSources), nSources, replace=False)
    sources = allSources[selected_idx].copy()

    # Normalize each source (peak)
    for i in range(nSources):
        s = sources[i]
        s = s / (np.max(np.abs(s)) + 1e-12)
        sources[i] = s
        if saveSignals:
            sf.write(f"audios/pooledSources/source_{i+1}.wav", s, fs, format="wav")

    # Scale to target SPL
    pRef = 20e-6  # Pa
    pTarget = pRef * 10 ** (sourcePower / 20)

    # Robust RMS estimate (avoid empty mask)
    pRmsNorm = []
    for i in range(nSources):
        s = sources[i]
        mask = np.abs(s) > 0.1
        if np.any(mask):
            rms = np.sqrt(np.mean(s[mask] ** 2))
        else:
            rms = np.sqrt(np.mean(s ** 2)) + 1e-12
        pRmsNorm.append(rms)
    pRmsNorm = np.array(pRmsNorm)

    scale = (pTarget / (pRmsNorm + 1e-12))[:, None]
    sources = scale * sources

    # ForestIR geometry
    fsize = 100.0
    fcenter = np.array([fsize / 2, fsize / 2, 1.5])
    treeDensity = 0.3
    nTrees = math.ceil(treeDensity * fsize * fsize)

    srcPos3D = np.column_stack((srcPos, np.full(srcPos.shape[0], 2.0)))
    micPos3D = np.column_stack((micPos, np.full(micPos.shape[0], 2.0)))

    # Generate mic signals per source and sum
    outputs = []
    for s in range(nSources):
        ir = generateSampleForestIR(
            fs,
            nTrees,
            srcPos3D[s] + fcenter,
            micPos3D + fcenter,
            forestRange_x=[0.0, fsize],
            forestRange_y=[0.0, fsize],
        )
        output = generateMicSignals(ir, sources[s])  # shape: [nSamples, nMics]
        outputs.append(output)

    # Sum outputs
    signals = np.zeros_like(outputs[0])
    for out in outputs:
        signals += out

    # Ensure shape is [nMics, nSamples]
    signals = signals.T

    # Keep to duration
    signals = signals[:, :max_samples]  # [nMics, nSamples]

    # Add uncorrelated Gaussian noise (SPL)
    pNoise = pRef * 10 ** (noisePower / 20)
    noise = np.random.normal(0, pNoise, signals.shape)
    # Compute SNR at each microphone
    snrs = []
    for i in range(nMics):
        signalPower = np.mean(signals[i] ** 2)
        noisePowerActual = np.mean(noise[i] ** 2)
        snr = 10 * np.log10(signalPower / noisePowerActual)
        snrs.append(snr)
    avg_snr = np.mean(snrs)
    print(f"Average SNR at microphones: {avg_snr:.2f} dB")
    signals = signals + noise

    # Normalize each mic channel and optionally save
    for m in range(nMics):
        signals[m] = signals[m] / (np.max(np.abs(signals[m])) + 1e-12)
        if saveSignals:
            sf.write(f"audios/micSignals/mic_{m+1}.wav", signals[m], fs, format="wav")

    return fs, micPos.astype(np.float32), signals.astype(np.float32), srcPos.astype(np.float32)


def load_sources_from_folder(
    sources_folder,
    fs,
    max_samples,
    pad_to_max=True,
    prefix="source",
    ext=".wav",
):
    sources = []
    for file in sorted(os.listdir(sources_folder)):
        if file.startswith(prefix) and file.endswith(ext):
            filepath = os.path.join(sources_folder, file)
            signal, _ = librosa.load(filepath, sr=fs)

            signal = signal[:max_samples]
            if pad_to_max and len(signal) < max_samples:
                signal = np.pad(signal, (0, max_samples - len(signal)))

            # force mono
            if signal.ndim == 2:
                signal = signal[:, 0]

            sources.append(signal)

    if len(sources) == 0:
        raise FileNotFoundError(f"No {prefix}*{ext} files found in {sources_folder}")

    return np.stack(sources)  # [nFiles, max_samples]


def simulation_setup_sparse_mics(nMics, nSources, environmentSize):
    """
    Generates microphone and source positions in 2D.
    """
    micPos = mic_positions_grid_fill(nMics, environmentSize)
    srcPos = np.random.rand(nSources, 2) * float(environmentSize)

    ### For testing, use fixed positions
    # srcPos = np.array([[27.88848278,  9.49126664],
    #        [ 9.71756435,  7.13680836],
    #        [17.03175087, 17.86634109],
    #        [28.93543559, 19.59531291]])
    # micPos = np.array([[0.5,0.5],[environmentSize-0.5, 0.5], [environmentSize//2, environmentSize-0.5]])
    # srcPos = np.array([[28, 15]])
    # micPos = np.array([[0,environmentSize/2],[environmentSize, environmentSize/2]])
    return micPos, srcPos


def mic_positions_grid_fill(nMics, environmentSize):
    """
    Places nMics on a near-square grid spanning [0, env]x[0, env],
    selecting points evenly across the grid.
    """
    env = float(environmentSize)
    if nMics <= 0:
        return np.zeros((0, 2), dtype=float)

    cols = int(np.ceil(np.sqrt(nMics)))
    rows = int(np.ceil(nMics / cols))

    xs = np.linspace(0.0, env, cols)
    ys = np.linspace(0.0, env, rows)

    grid = np.array([(x, y) for y in ys for x in xs], dtype=float)

    idx = np.round(np.linspace(0, len(grid) - 1, nMics)).astype(int)
    idx = np.unique(idx)

    if len(idx) < nMics:
        remaining = np.setdiff1d(np.arange(len(grid)), idx, assume_unique=False)
        need = nMics - len(idx)
        extra = remaining[np.round(np.linspace(0, len(remaining) - 1, need)).astype(int)]
        idx = np.concatenate([idx, extra])

    return grid[idx]
