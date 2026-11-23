"""Student 3: Audio analysis and sinusoidal interference.

Loads a WAV audio file, adds a 50 Hz sinusoidal interference tone, saves the
noisy audio as noisy.wav for Student 4, and generates waveform and spectrum plots.

Outputs:
  - noisy.wav            clean audio + 50 Hz interference tone
  - audio_analysis.png   clean vs. noisy waveforms and full spectra (dB)
  - spectrum_zoom.png    low-frequency spectrum (0-200 Hz) highlighting 50 Hz
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


# Input and output file paths.
AUDIO_PATH = Path("audio.wav")
NOISY_WAV_PATH = Path("noisy.wav")
MAIN_FIGURE_PATH = Path("audio_analysis.png")
ZOOM_FIGURE_PATH = Path("spectrum_zoom.png")

# Signal-processing parameters.
INTERFERENCE_FREQUENCY_HZ = 50   # Power-line interference frequency (50 or 60 Hz)
INTERFERENCE_LEVEL = 0.20        # Tone amplitude as a fraction of the clean peak
SEARCH_BAND_HZ = (40, 70)        # Band searched to identify mains interference
ZOOM_MAX_HZ = 200                # Upper frequency limit for the focused plot


def to_float_audio(audio):
    """Convert WAV integer or float samples to float64 in approximately [-1, 1]."""
    if audio.dtype == np.uint8:
        # 8-bit WAV files are unsigned with silence at 128.
        return (audio.astype(np.float64) - 128.0) / 128.0
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        return audio.astype(np.float64) / max(abs(info.min), info.max)
    return audio.astype(np.float64)


def save_wav(path, signal, sample_rate):
    """Save a mono float signal as 16-bit PCM WAV, scaling only if needed to avoid clipping."""
    peak = np.max(np.abs(signal))
    output = (0.99 * signal / peak) if peak > 0.99 else signal
    wavfile.write(path, sample_rate, np.round(output * 32767).astype(np.int16))


def magnitude_spectrum(signal, sample_rate):
    """Return the single-sided amplitude spectrum for a real-valued signal."""
    count = len(signal)
    spectrum = np.abs(np.fft.rfft(signal)) / count
    # Double interior bins to fold negative-frequency energy onto positive frequencies.
    # DC (index 0) and Nyquist (last bin for even N) have no mirror bin.
    if count % 2 == 0:
        spectrum[1:-1] *= 2
    else:
        spectrum[1:] *= 2
    frequencies = np.fft.rfftfreq(count, d=1 / sample_rate)
    return frequencies, spectrum


def to_db(magnitude):
    """Convert linear magnitude to decibels (dB), avoiding log10(0)."""
    return 20.0 * np.log10(np.maximum(magnitude, 1e-12))


def find_peak_in_band(frequencies, magnitude, low_hz, high_hz):
    """Return (frequency, magnitude) of the strongest peak within [low_hz, high_hz]."""
    band_indices = np.where((frequencies >= low_hz) & (frequencies <= high_hz))[0]
    if len(band_indices) == 0:
        raise ValueError(
            f"No frequency bins between {low_hz} and {high_hz} Hz; audio may be too short."
        )
    best = band_indices[np.argmax(magnitude[band_indices])]
    return frequencies[best], magnitude[best]


def plot_analysis(time, clean, noisy, freqs, clean_db, noisy_db, sample_rate, peak_freq, peak_db):
    """Create the 2x2 figure comparing clean and noisy waveforms and full spectra."""
    duration = len(clean) / sample_rate

    # Shared axis limits make visual comparisons between clean and noisy fair.
    wave_limit = 1.08 * max(np.max(np.abs(clean)), np.max(np.abs(noisy)))
    spec_top = max(np.max(clean_db), np.max(noisy_db)) + 5.0
    spec_bottom = max(spec_top - 105.0, -120.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Top row: time-domain waveforms.
    axes[0, 0].plot(time, clean, linewidth=0.6, color="tab:blue")
    axes[0, 0].set_title("Clean Audio Waveform")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_xlim(0, duration)
    axes[0, 0].set_ylim(-wave_limit, wave_limit)

    axes[0, 1].plot(time, noisy, linewidth=0.6, color="tab:orange")
    axes[0, 1].set_title(f"Noisy Waveform (+{INTERFERENCE_FREQUENCY_HZ} Hz Tone)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].set_xlim(0, duration)
    axes[0, 1].set_ylim(-wave_limit, wave_limit)

    # Bottom row: full frequency spectra in dB (DC excluded).
    axes[1, 0].plot(freqs, clean_db, linewidth=0.7, color="tab:blue")
    axes[1, 0].set_title("Clean Audio Magnitude Spectrum")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Magnitude (dB)")
    axes[1, 0].set_xlim(0, sample_rate / 2)
    axes[1, 0].set_ylim(spec_bottom, spec_top)

    axes[1, 1].plot(freqs, noisy_db, linewidth=0.7, color="tab:orange")
    axes[1, 1].axvline(INTERFERENCE_FREQUENCY_HZ, color="red", linestyle="--", alpha=0.45)
    axes[1, 1].plot(
        peak_freq, peak_db, "ro", markersize=5,
        label=f"Interference peak: {peak_freq:.1f} Hz ({peak_db:.1f} dB)",
    )
    axes[1, 1].set_title("Noisy Audio Magnitude Spectrum")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Magnitude (dB)")
    axes[1, 1].set_xlim(0, sample_rate / 2)
    axes[1, 1].set_ylim(spec_bottom, spec_top)
    axes[1, 1].legend(loc="upper right")

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)

    fig.suptitle(
        f"Audio Analysis | Fs = {sample_rate} Hz | Duration = {duration:.2f} s",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(MAIN_FIGURE_PATH, dpi=150)


def plot_zoomed_spectrum(freqs, clean_db, noisy_db, sample_rate, peak_freq, peak_db):
    """Create the focused 0-200 Hz spectrum highlighting the added 50 Hz interference."""
    zoom_max = min(ZOOM_MAX_HZ, sample_rate / 2)
    mask = freqs <= zoom_max

    spec_top = max(np.max(clean_db[mask]), np.max(noisy_db[mask])) + 6.0
    spec_bottom = max(spec_top - 105.0, -120.0)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(freqs[mask], noisy_db[mask], linewidth=1.1, color="tab:orange", label="Noisy")
    ax.plot(freqs[mask], clean_db[mask], linewidth=1.0, color="tab:blue", alpha=0.85, label="Clean")
    ax.axvline(INTERFERENCE_FREQUENCY_HZ, color="red", linestyle="--", alpha=0.4)
    ax.plot(peak_freq, peak_db, "ro", markersize=6, label=f"Detected peak: {peak_freq:.1f} Hz")
    ax.annotate(
        f"{peak_freq:.1f} Hz ({peak_db:.1f} dB)",
        xy=(peak_freq, peak_db),
        xytext=(peak_freq + 14, peak_db - 4),
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.0},
        fontsize=9.5,
        color="darkred",
    )

    ax.set_title(f"Low-Frequency Spectrum Comparison (0-{zoom_max:g} Hz, DC Excluded)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(0, zoom_max)
    ax.set_ylim(spec_bottom, spec_top)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ZOOM_FIGURE_PATH, dpi=150)


def main():
    # 1) Load WAV audio, convert to float in [-1, 1], and convert stereo to mono.
    if not AUDIO_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find WAV file: {AUDIO_PATH.resolve()}\n"
            "Place audio.wav beside this script or update AUDIO_PATH."
        )

    sample_rate, raw_audio = wavfile.read(AUDIO_PATH)
    clean = to_float_audio(raw_audio)
    if clean.ndim == 2:
        clean = clean.mean(axis=1)

    if len(clean) == 0 or np.max(np.abs(clean)) == 0:
        raise ValueError("The WAV file is empty or completely silent.")
    if not 0 < INTERFERENCE_FREQUENCY_HZ < sample_rate / 2:
        raise ValueError(
            f"Interference frequency ({INTERFERENCE_FREQUENCY_HZ} Hz) must be "
            f"below the Nyquist frequency ({sample_rate / 2:g} Hz)."
        )

    # 2) Generate and add sinusoidal interference: A * sin(2 * pi * f0 * n / fs).
    sample_indices = np.arange(len(clean))
    tone_amplitude = INTERFERENCE_LEVEL * np.max(np.abs(clean))
    interference = tone_amplitude * np.sin(
        2.0 * np.pi * INTERFERENCE_FREQUENCY_HZ * sample_indices / sample_rate
    )
    noisy = clean + interference

    # Save noisy audio for Student 4 without unnecessary volume scaling.
    save_wav(NOISY_WAV_PATH, noisy, sample_rate)

    # 3) Compute single-sided magnitude spectra and identify the interference peak.
    duration = len(clean) / sample_rate
    time = sample_indices / sample_rate
    freqs, clean_mag = magnitude_spectrum(clean, sample_rate)
    _, noisy_mag = magnitude_spectrum(noisy, sample_rate)

    band_low, band_high = SEARCH_BAND_HZ
    peak_freq, peak_mag = find_peak_in_band(freqs, noisy_mag, band_low, band_high)
    diff_freq = freqs[np.argmax(noisy_mag - clean_mag)]

    # Exclude DC (0 Hz, index 0) when plotting so any DC offset does not dominate.
    plot_freqs = freqs[1:]
    clean_db = to_db(clean_mag[1:])
    noisy_db = to_db(noisy_mag[1:])
    peak_db = float(to_db(peak_mag))

    # 4) Generate presentation-ready figures.
    plot_analysis(time, clean, noisy, plot_freqs, clean_db, noisy_db, sample_rate, peak_freq, peak_db)
    plot_zoomed_spectrum(plot_freqs, clean_db, noisy_db, sample_rate, peak_freq, peak_db)

    # 5) Print summary.
    print(f"Sample rate: {sample_rate} Hz (Nyquist: {sample_rate / 2:g} Hz)")
    print(f"Duration: {duration:.2f} s | Frequency resolution: {1 / duration:.4f} Hz/bin")
    print(f"Added interference: {INTERFERENCE_FREQUENCY_HZ} Hz (amplitude = {tone_amplitude:.4f})")
    print(
        f"Detected peak in {band_low}-{band_high} Hz band: "
        f"{peak_freq:.2f} Hz (magnitude = {peak_mag:.5f}, {peak_db:.2f} dB)"
    )
    print(f"Cross-check largest (noisy - clean) difference at: {diff_freq:.2f} Hz")
    print(f"Saved: {NOISY_WAV_PATH}, {MAIN_FIGURE_PATH}, {ZOOM_FIGURE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
