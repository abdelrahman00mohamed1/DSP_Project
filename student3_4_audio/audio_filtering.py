"""Student 4: Audio filtering with a second-order IIR Notch Filter.

Loads noisy.wav from Student 3, calculates the coefficients of a second-order
IIR Notch Filter at 50 Hz, applies it using lfilter, saves filtered.wav, and
plots the before/after spectra and filter magnitude response.

Outputs:
  - filtered.wav              audio after applying the 50 Hz Notch Filter
  - filtering_comparison.png  before/after spectra and filter magnitude response
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import freqz, lfilter


# File paths and filter parameters.
NOISY_WAV_PATH = Path("noisy.wav")
FILTERED_WAV_PATH = Path("filtered.wav")
COMPARISON_FIGURE_PATH = Path("filtering_comparison.png")

NOTCH_FREQUENCY_HZ = 50   # Must match the interference frequency from Student 3
NOTCH_BANDWIDTH_HZ = 10   # Desired -3 dB notch width (Hz); used to compute r
ZOOM_MAX_HZ = 200         # Upper frequency limit for focused plots

# Quiet light-paper plot theme (ink neutral ramp + vermilion/teal before/after).
BG_COLOR = "#f5f7fa"
NOISY_COLOR = "#c84b31"
FILTERED_COLOR = "#0f766e"
RESPONSE_COLOR = "#335c81"
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
        raise FileNotFoundError(
            f"Could not find noisy WAV file: {path.resolve()}\n"
            "Run audio_analysis.py first so noisy.wav is in the same folder."
        )

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
    if len(signal) == 0:
        raise ValueError("The WAV file contains no audio samples.")
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
    if count % 2 == 0:
        spectrum[1:-1] *= 2
    else:
        spectrum[1:] *= 2
    frequencies = np.fft.rfftfreq(count, d=1 / sample_rate)
    return frequencies, spectrum


def to_db(magnitude):
    """Convert linear magnitude to decibels (dB), avoiding log10(0)."""
    return 20.0 * np.log10(np.maximum(magnitude, 1e-12))


def calculate_notch_coefficients(notch_hz, sample_rate, pole_radius):
    """Calculate numerator (b) and denominator (a) for a 2nd-order IIR Notch Filter.

    Transfer function:
        H(z) = (1 - 2*cos(w0)*z^-1 + z^-2) / (1 - 2*r*cos(w0)*z^-1 + r^2*z^-2)
    where w0 = 2 * pi * f0 / fs.
    """
    w0 = 2.0 * np.pi * notch_hz / sample_rate
    b = np.array([1.0, -2.0 * np.cos(w0), 1.0])                        # Zeros on unit circle at +/- w0
    a = np.array([1.0, -2.0 * pole_radius * np.cos(w0), pole_radius**2])  # Poles at radius r < 1
    return w0, b, a


def plot_filtering_comparison(freqs, noisy_mag, filtered_mag, b, a, sample_rate, duration):
    """Save the 2x2 comparison figure: full spectra, 0-200 Hz overlay, and |H(f)|."""
    # Exclude DC (0 Hz, index 0) so any DC offset does not dominate the plots.
    plot_freqs = freqs[1:]
    noisy_db = to_db(noisy_mag[1:])
    filtered_db = to_db(filtered_mag[1:])

    notch_idx = np.argmin(np.abs(plot_freqs - NOTCH_FREQUENCY_HZ))
    measured_freq = plot_freqs[notch_idx]
    before_db = float(noisy_db[notch_idx])
    after_db = float(filtered_db[notch_idx])
    reduction_db = after_db - before_db

    zoom_max = min(ZOOM_MAX_HZ, sample_rate / 2)
    zoom_mask = plot_freqs <= zoom_max

    # Shared y-axis limits make before/after comparisons directly readable.
    spec_top = max(np.max(noisy_db), np.max(filtered_db)) + 6.0
    spec_bottom = max(spec_top - 105.0, -120.0)

    # Theoretical filter magnitude response over 0 to zoom_max Hz.
    resp_freqs = np.linspace(0.0, zoom_max, 2000)
    _, h_response = freqz(b, a, worN=resp_freqs, fs=sample_rate)
    resp_db = to_db(np.abs(h_response))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # Top-left: full spectrum before filtering.
    ax = axes[0, 0]
    ax.plot(plot_freqs, noisy_db, linewidth=0.65, color=NOISY_COLOR)
    ax.axvline(NOTCH_FREQUENCY_HZ, color=MUTED_COLOR, linestyle="--", linewidth=0.9, alpha=0.5)
    ax.plot(measured_freq, before_db, "o", color=NOISY_COLOR, markeredgecolor="#ffffff", markeredgewidth=1.2, markersize=6,
            label=f"{measured_freq:.1f} Hz ({before_db:.1f} dB)")
    ax.set(xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", xlim=(0, sample_rate / 2), ylim=(spec_bottom, spec_top))
    ax.set_title("Full spectrum before filtering (noisy)", loc="left", fontsize=11, pad=8)
    ax.legend(loc="upper right", fontsize=9)

    # Top-right: full spectrum after filtering (same y-scale).
    ax = axes[0, 1]
    ax.plot(plot_freqs, filtered_db, linewidth=0.65, color=FILTERED_COLOR)
    ax.axvline(NOTCH_FREQUENCY_HZ, color=MUTED_COLOR, linestyle="--", linewidth=0.9, alpha=0.5)
    ax.plot(measured_freq, after_db, "o", color=FILTERED_COLOR, markeredgecolor="#ffffff", markeredgewidth=1.2, markersize=6,
            label=f"{measured_freq:.1f} Hz ({after_db:.1f} dB)")
    ax.set(xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", xlim=(0, sample_rate / 2), ylim=(spec_bottom, spec_top))
    ax.set_title("Full spectrum after filtering (notch filtered)", loc="left", fontsize=11, pad=8)
    ax.legend(loc="upper right", fontsize=9)

    # Bottom-left: focused 0-200 Hz before vs. after overlay.
    ax = axes[1, 0]
    ax.plot(plot_freqs[zoom_mask], noisy_db[zoom_mask], linewidth=1.05, color=NOISY_COLOR, alpha=0.85, label="Before (noisy)")
    ax.plot(plot_freqs[zoom_mask], filtered_db[zoom_mask], linewidth=1.0, color=FILTERED_COLOR, alpha=0.95, label="After (filtered)")
    ax.axvline(NOTCH_FREQUENCY_HZ, color=MUTED_COLOR, linestyle="--", linewidth=0.9, alpha=0.45)
    ax.plot(measured_freq, before_db, "o", color=NOISY_COLOR, markeredgecolor="#ffffff", markeredgewidth=1.2, markersize=6.5)
    ax.plot(measured_freq, after_db, "o", color=FILTERED_COLOR, markeredgecolor="#ffffff", markeredgewidth=1.2, markersize=6.5)
    ax.annotate(
        f"Attenuation: {reduction_db:.1f} dB",
        xy=(measured_freq, after_db), xytext=(measured_freq + 20, (before_db + after_db) / 2 + 4),
        arrowprops={"arrowstyle": "->", "color": FILTERED_COLOR, "lw": 1.1},
        fontsize=9.3, color=FILTERED_COLOR, family="monospace",
    )
    ax.set(xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", xlim=(0, zoom_max), ylim=(spec_bottom, spec_top))
    ax.set_title(f"Low-frequency comparison (0–{zoom_max:g} Hz)", loc="left", fontsize=11, pad=8)
    ax.legend(loc="lower right", fontsize=9)

    # Bottom-right: theoretical Notch Filter magnitude response |H(f)|.
    ax = axes[1, 1]
    ax.plot(resp_freqs, resp_db, linewidth=1.4, color=RESPONSE_COLOR, label="|H(f)|")
    ax.axvline(NOTCH_FREQUENCY_HZ, color=NOISY_COLOR, linestyle="--", linewidth=0.9, alpha=0.55, label=f"Notch: {NOTCH_FREQUENCY_HZ} Hz")
    ax.axhline(-3.0, color=MUTED_COLOR, linestyle=":", linewidth=1.0, label=f"-3 dB line (BW ≈ {NOTCH_BANDWIDTH_HZ} Hz)")
    ax.set(xlabel="Frequency (Hz)", ylabel="Gain (dB)", xlim=(0, zoom_max), ylim=(-60, 5))
    ax.set_title(f"2nd-order notch filter response (0–{zoom_max:g} Hz)", loc="left", fontsize=11, pad=8)
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Second-order IIR notch filtering comparison", x=0.04, y=0.975, ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.04, 0.938,
        f"Fs = {sample_rate:,} Hz  ·  Duration = {duration:.2f} s  ·  f0 = {NOTCH_FREQUENCY_HZ} Hz  ·  BW = {NOTCH_BANDWIDTH_HZ} Hz  ·  DC excluded",
        fontsize=9.5, color=MUTED_COLOR, family="monospace",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.915), h_pad=2.2, w_pad=2.0)
    fig.savefig(COMPARISON_FIGURE_PATH, dpi=150)


def main():
    # 1) Load noisy.wav and validate parameters.
    sample_rate, noisy = load_mono_audio(NOISY_WAV_PATH)
    if not 0 < NOTCH_FREQUENCY_HZ < sample_rate / 2:
        raise ValueError(
            f"Notch frequency ({NOTCH_FREQUENCY_HZ} Hz) must be below Nyquist ({sample_rate / 2:g} Hz)."
        )

    # Pole radius from desired -3 dB bandwidth: BW ≈ (1 - r) * fs / pi.
    pole_radius = 1.0 - np.pi * NOTCH_BANDWIDTH_HZ / sample_rate
    if not 0 < pole_radius < 1:
        raise ValueError(f"Pole radius r must be in (0, 1) for stability; got {pole_radius:.6f}.")

    # 2) Calculate second-order Notch Filter coefficients and apply the filter.
    w0, b, a = calculate_notch_coefficients(NOTCH_FREQUENCY_HZ, sample_rate, pole_radius)
    filtered = lfilter(b, a, noisy)
    save_wav(FILTERED_WAV_PATH, filtered, sample_rate)

    # 3) Compute spectra before/after filtering and measure attenuation at 50 Hz.
    freqs, noisy_mag = magnitude_spectrum(noisy, sample_rate)
    _, filtered_mag = magnitude_spectrum(filtered, sample_rate)

    notch_bin = np.argmin(np.abs(freqs - NOTCH_FREQUENCY_HZ))
    measured_freq = freqs[notch_bin]
    mag_before, mag_after = noisy_mag[notch_bin], filtered_mag[notch_bin]
    reduction_percent = (1.0 - mag_after / mag_before) * 100.0 if mag_before > 0 else 0.0
    before_db, after_db = float(to_db(mag_before)), float(to_db(mag_after))

    # 4) Plot and report results.
    plot_filtering_comparison(freqs, noisy_mag, filtered_mag, b, a, sample_rate, len(noisy) / sample_rate)

    print(f"Sample rate (fs): {sample_rate} Hz")
    print(f"Notch frequency (f0): {NOTCH_FREQUENCY_HZ} Hz | Bandwidth (BW): {NOTCH_BANDWIDTH_HZ} Hz")
    print(f"Digital angular frequency (w0): {w0:.6f} rad/sample | Pole radius (r): {pole_radius:.6f}")
    print(f"Numerator coefficients   b = [{b[0]:.6f}, {b[1]:.6f}, {b[2]:.6f}]")
    print(f"Denominator coefficients a = [{a[0]:.6f}, {a[1]:.6f}, {a[2]:.6f}]")
    print(f"Magnitude at {measured_freq:.2f} Hz before: {mag_before:.6f} ({before_db:.2f} dB)")
    print(f"Magnitude at {measured_freq:.2f} Hz after:  {mag_after:.6f} ({after_db:.2f} dB)")
    print(f"Peak reduction at ~{NOTCH_FREQUENCY_HZ} Hz: {reduction_percent:.2f}% ({after_db - before_db:.2f} dB)")
    print(f"Saved: {FILTERED_WAV_PATH}, {COMPARISON_FIGURE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
