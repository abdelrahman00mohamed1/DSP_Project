"""Student 3: Audio analysis and sinusoidal interference.

Loads audio.wav, adds a 50 Hz sinusoidal interference tone, saves noisy.wav for
Student 4, and generates waveform and frequency-spectrum plots.

Outputs:
  - noisy.wav            clean audio + 50 Hz interference tone
  - audio_analysis.png   clean vs. noisy waveforms and full spectra (dB)
  - spectrum_zoom.png    low-frequency spectrum (0-200 Hz) highlighting 50 Hz
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


# File paths and DSP parameters.
AUDIO_PATH = Path("audio.wav")
NOISY_WAV_PATH = Path("noisy.wav")
MAIN_FIGURE_PATH = Path("audio_analysis.png")
ZOOM_FIGURE_PATH = Path("spectrum_zoom.png")

INTERFERENCE_FREQUENCY_HZ = 50   # Power-line interference frequency (50 or 60 Hz)
INTERFERENCE_LEVEL = 0.20        # Tone amplitude as a fraction of the clean peak
SEARCH_BAND_HZ = (40, 70)        # Band searched to identify mains interference
ZOOM_MAX_HZ = 200                # Upper frequency limit for the focused plot

# Quiet light-paper plot theme (ink neutral ramp + vermilion accent for interference).
BG_COLOR = "#f5f7fa"
CLEAN_COLOR = "#335c81"
NOISY_COLOR = "#c84b31"
MUTED_COLOR = "#6b7487"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cbd5e1",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.7,
    "axes.titlecolor": "#111827",
    "axes.labelcolor": "#4b5565",
    "xtick.color": MUTED_COLOR,
    "ytick.color": MUTED_COLOR,
    "legend.frameon": False,
})


def load_mono_audio(path):
    """Load a WAV file as mono float64 samples in approximately [-1, 1]."""
    if not path.is_file():
        raise FileNotFoundError(f"Could not find WAV file: {path.resolve()}")

    sample_rate, raw = wavfile.read(path)
    if raw.dtype == np.uint8:
        signal = (raw.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        signal = raw.astype(np.float64) / max(abs(info.min), info.max)
    else:
        signal = raw.astype(np.float64)

    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if len(signal) == 0 or np.max(np.abs(signal)) == 0:
        raise ValueError("The WAV file is empty or completely silent.")
    return sample_rate, signal


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
    indices = np.where((frequencies >= low_hz) & (frequencies <= high_hz))[0]
    if len(indices) == 0:
        raise ValueError(f"No frequency bins in [{low_hz}, {high_hz}] Hz; audio may be too short.")
    best = indices[np.argmax(magnitude[indices])]
    return frequencies[best], magnitude[best]


def plot_analysis(clean, noisy, freqs, clean_mag, noisy_mag, sample_rate, peak_freq, peak_mag):
    """Save the 2x2 waveform/spectrum overview and the focused 0-200 Hz spectrum."""
    duration = len(clean) / sample_rate
    time = np.arange(len(clean)) / sample_rate

    # Exclude DC (0 Hz, index 0) when plotting so any DC offset does not dominate.
    plot_freqs = freqs[1:]
    clean_db = to_db(clean_mag[1:])
    noisy_db = to_db(noisy_mag[1:])
    peak_db = float(to_db(peak_mag))

    # Shared axis limits keep clean vs. noisy visual comparisons fair.
    wave_lim = 1.08 * max(np.max(np.abs(clean)), np.max(np.abs(noisy)))
    spec_top = max(np.max(clean_db), np.max(noisy_db)) + 6.0
    spec_bottom = max(spec_top - 105.0, -120.0)

    # Figure 1: 2x2 waveforms and full spectra.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(time, clean, linewidth=0.55, color=CLEAN_COLOR)
    axes[0, 0].set(xlabel="Time (s)", ylabel="Amplitude", xlim=(0, duration), ylim=(-wave_lim, wave_lim))
    axes[0, 0].set_title("Clean audio waveform", loc="left", fontsize=11, pad=8)

    axes[0, 1].plot(time, noisy, linewidth=0.55, color=NOISY_COLOR)
    axes[0, 1].set(xlabel="Time (s)", ylabel="Amplitude", xlim=(0, duration), ylim=(-wave_lim, wave_lim))
    axes[0, 1].set_title(f"Noisy waveform (+{INTERFERENCE_FREQUENCY_HZ} Hz tone)", loc="left", fontsize=11, pad=8)

    axes[1, 0].plot(plot_freqs, clean_db, linewidth=0.65, color=CLEAN_COLOR)
    axes[1, 0].set(xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", xlim=(0, sample_rate / 2), ylim=(spec_bottom, spec_top))
    axes[1, 0].set_title("Clean audio spectrum", loc="left", fontsize=11, pad=8)

    axes[1, 1].plot(plot_freqs, noisy_db, linewidth=0.65, color=NOISY_COLOR)
    axes[1, 1].axvline(INTERFERENCE_FREQUENCY_HZ, color=MUTED_COLOR, linestyle="--", linewidth=0.9, alpha=0.5)
    axes[1, 1].plot(
        peak_freq, peak_db, "o", color=NOISY_COLOR, markeredgecolor="#ffffff", markeredgewidth=1.2, markersize=6,
        label=f"Interference peak: {peak_freq:.1f} Hz ({peak_db:.1f} dB)",
    )
    axes[1, 1].set(xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", xlim=(0, sample_rate / 2), ylim=(spec_bottom, spec_top))
    axes[1, 1].set_title("Noisy audio spectrum", loc="left", fontsize=11, pad=8)
    axes[1, 1].legend(loc="upper right", fontsize=9)

    fig.suptitle("Audio analysis and 50 Hz interference", x=0.04, y=0.975, ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.04, 0.935,
        f"Fs = {sample_rate:,} Hz  ·  Duration = {duration:.2f} s  ·  Resolution = {1 / duration:.3f} Hz/bin  ·  DC excluded",
        fontsize=9.5, color=MUTED_COLOR, family="monospace",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2.2, w_pad=2.0)
    fig.savefig(MAIN_FIGURE_PATH, dpi=150)

    # Figure 2: focused 0-200 Hz spectrum highlighting the interference peak.
    zoom_max = min(ZOOM_MAX_HZ, sample_rate / 2)
    mask = plot_freqs <= zoom_max
    zoom_top = max(np.max(clean_db[mask]), np.max(noisy_db[mask])) + 8.0
    zoom_bottom = max(zoom_top - 105.0, -120.0)

    fig_zoom, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.plot(plot_freqs[mask], noisy_db[mask], linewidth=1.1, color=NOISY_COLOR, alpha=0.95, label=f"Noisy (+{INTERFERENCE_FREQUENCY_HZ} Hz)")
    ax.plot(plot_freqs[mask], clean_db[mask], linewidth=0.95, color=CLEAN_COLOR, alpha=0.85, label="Clean")
    ax.axvline(INTERFERENCE_FREQUENCY_HZ, color=MUTED_COLOR, linestyle="--", linewidth=0.9, alpha=0.45)
    ax.plot(peak_freq, peak_db, "o", color=NOISY_COLOR, markeredgecolor="#ffffff", markeredgewidth=1.4, markersize=7)
    ax.annotate(
        f"{peak_freq:.1f} Hz  ({peak_db:.1f} dB)",
        xy=(peak_freq, peak_db), xytext=(peak_freq + 14, peak_db - 2.5),
        arrowprops={"arrowstyle": "->", "color": NOISY_COLOR, "lw": 1.1},
        fontsize=9.5, color=NOISY_COLOR, family="monospace",
    )
    ax.set(xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", xlim=(0, zoom_max), ylim=(zoom_bottom, zoom_top))
    ax.legend(loc="lower right", fontsize=9.5)

    fig_zoom.suptitle(f"Low-frequency spectrum comparison (0–{zoom_max:g} Hz)", x=0.06, y=0.96, ha="left", fontsize=13, fontweight="bold")
    fig_zoom.text(0.06, 0.905, f"Detected interference peak at {peak_freq:.2f} Hz  ·  DC excluded", fontsize=9.2, color=MUTED_COLOR, family="monospace")
    fig_zoom.tight_layout(rect=(0, 0, 1, 0.88))
    fig_zoom.savefig(ZOOM_FIGURE_PATH, dpi=150)


def main():
    # 1) Load clean audio and validate interference frequency against Nyquist.
    sample_rate, clean = load_mono_audio(AUDIO_PATH)
    if not 0 < INTERFERENCE_FREQUENCY_HZ < sample_rate / 2:
        raise ValueError(
            f"Interference frequency ({INTERFERENCE_FREQUENCY_HZ} Hz) must be below Nyquist ({sample_rate / 2:g} Hz)."
        )

    # 2) Generate and add sinusoidal interference: A * sin(2 * pi * f0 * n / fs).
    sample_indices = np.arange(len(clean))
    tone_amplitude = INTERFERENCE_LEVEL * np.max(np.abs(clean))
    interference = tone_amplitude * np.sin(2.0 * np.pi * INTERFERENCE_FREQUENCY_HZ * sample_indices / sample_rate)
    noisy = clean + interference
    save_wav(NOISY_WAV_PATH, noisy, sample_rate)

    # 3) Compute spectra and identify the interference frequency.
    freqs, clean_mag = magnitude_spectrum(clean, sample_rate)
    _, noisy_mag = magnitude_spectrum(noisy, sample_rate)

    band_low, band_high = SEARCH_BAND_HZ
    peak_freq, peak_mag = find_peak_in_band(freqs, noisy_mag, band_low, band_high)
    diff_freq = freqs[np.argmax(noisy_mag - clean_mag)]

    # 4) Plot and report results.
    plot_analysis(clean, noisy, freqs, clean_mag, noisy_mag, sample_rate, peak_freq, peak_mag)

    duration = len(clean) / sample_rate
    print(f"Sample rate: {sample_rate} Hz (Nyquist: {sample_rate / 2:g} Hz)")
    print(f"Duration: {duration:.2f} s | Frequency resolution: {1 / duration:.4f} Hz/bin")
    print(f"Added interference: {INTERFERENCE_FREQUENCY_HZ} Hz (amplitude = {tone_amplitude:.4f})")
    print(f"Detected peak in {band_low}-{band_high} Hz band: {peak_freq:.2f} Hz (magnitude = {peak_mag:.5f}, {to_db(peak_mag):.2f} dB)")
    print(f"Cross-check largest (noisy - clean) difference at: {diff_freq:.2f} Hz")
    print(f"Saved: {NOISY_WAV_PATH}, {MAIN_FIGURE_PATH}, {ZOOM_FIGURE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
