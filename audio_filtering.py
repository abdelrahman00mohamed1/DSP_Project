"""Student 4: Audio filtering with a second-order IIR Notch Filter.

Loads noisy.wav from Student 3, calculates the coefficients of a second-order
IIR Notch Filter at 50 Hz, applies it using lfilter, saves filtered.wav, and
plots the before/after spectra and filter frequency response.

Outputs:
  - filtered.wav              audio after applying the 50 Hz Notch Filter
  - filtering_comparison.png  before/after spectra and filter magnitude response
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import freqz, lfilter


# Input and output file paths.
NOISY_WAV_PATH = Path("noisy.wav")
FILTERED_WAV_PATH = Path("filtered.wav")
COMPARISON_FIGURE_PATH = Path("filtering_comparison.png")

# Filter and plotting parameters.
NOTCH_FREQUENCY_HZ = 50   # Must match the interference frequency from Student 3
NOTCH_BANDWIDTH_HZ = 10   # Desired -3 dB notch width (Hz); used to compute r
ZOOM_MAX_HZ = 200         # Upper frequency limit for focused plots


def to_float_audio(audio):
    """Convert WAV integer or float samples to float64 in approximately [-1, 1]."""
    if audio.dtype == np.uint8:
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
    """Calculate numerator (b) and denominator (a) coefficients for a 2nd-order IIR Notch Filter.

    Transfer function:
        H(z) = (1 - 2*cos(w0)*z^-1 + z^-2) / (1 - 2*r*cos(w0)*z^-1 + r^2*z^-2)
    where w0 = 2 * pi * f0 / fs.
    """
    w0 = 2.0 * np.pi * notch_hz / sample_rate

    # Zeros on the unit circle at angle +/- w0 force |H(w0)| = 0.
    b = np.array([1.0, -2.0 * np.cos(w0), 1.0])

    # Poles at radius r < 1 and angle +/- w0 keep the notch narrow and stable.
    a = np.array([1.0, -2.0 * pole_radius * np.cos(w0), pole_radius**2])

    return w0, b, a


def plot_filtering_comparison(
    freqs, noisy_db, filtered_db, b, a, sample_rate, duration,
    measured_freq, before_db, after_db, reduction_db,
):
    """Create a 2x2 figure: full spectra (before/after), 0-200 Hz overlay, and |H(f)|."""
    zoom_max = min(ZOOM_MAX_HZ, sample_rate / 2)
    zoom_mask = freqs <= zoom_max

    # Shared y-axis limits make before/after comparisons directly readable.
    spec_top = max(np.max(noisy_db), np.max(filtered_db)) + 5.0
    spec_bottom = max(spec_top - 105.0, -120.0)

    # Theoretical filter frequency response over 0 to zoom_max Hz.
    resp_freqs = np.linspace(0.0, zoom_max, 2000)
    _, h_response = freqz(b, a, worN=resp_freqs, fs=sample_rate)
    resp_db = to_db(np.abs(h_response))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # Top-left: full spectrum before filtering.
    ax = axes[0, 0]
    ax.plot(freqs, noisy_db, linewidth=0.7, color="tab:orange")
    ax.axvline(NOTCH_FREQUENCY_HZ, color="red", linestyle="--", alpha=0.45)
    ax.plot(
        measured_freq, before_db, "ro", markersize=5,
        label=f"{measured_freq:.1f} Hz ({before_db:.1f} dB)",
    )
    ax.set_title("Full Spectrum Before Filtering (Noisy)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(0, sample_rate / 2)
    ax.set_ylim(spec_bottom, spec_top)
    ax.legend(loc="upper right")

    # Top-right: full spectrum after filtering (same y-scale as top-left).
    ax = axes[0, 1]
    ax.plot(freqs, filtered_db, linewidth=0.7, color="tab:green")
    ax.axvline(NOTCH_FREQUENCY_HZ, color="red", linestyle="--", alpha=0.45)
    ax.plot(
        measured_freq, after_db, "go", markersize=5,
        label=f"{measured_freq:.1f} Hz ({after_db:.1f} dB)",
    )
    ax.set_title("Full Spectrum After Filtering (Notch Filtered)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(0, sample_rate / 2)
    ax.set_ylim(spec_bottom, spec_top)
    ax.legend(loc="upper right")

    # Bottom-left: focused 0-200 Hz before vs. after overlay.
    ax = axes[1, 0]
    ax.plot(freqs[zoom_mask], noisy_db[zoom_mask], linewidth=1.1, color="tab:orange", label="Before (Noisy)")
    ax.plot(freqs[zoom_mask], filtered_db[zoom_mask], linewidth=1.0, color="tab:green", alpha=0.9, label="After (Filtered)")
    ax.axvline(NOTCH_FREQUENCY_HZ, color="red", linestyle="--", alpha=0.4)
    ax.plot(measured_freq, before_db, "o", color="tab:orange", markersize=6)
    ax.plot(measured_freq, after_db, "o", color="tab:green", markersize=6)
    ax.annotate(
        f"Reduction: {reduction_db:.1f} dB",
        xy=(measured_freq, after_db),
        xytext=(measured_freq + 18, (before_db + after_db) / 2),
        arrowprops={"arrowstyle": "->", "color": "darkgreen", "lw": 1.0},
        fontsize=9.5,
        color="darkgreen",
    )
    ax.set_title(f"Focused Spectrum Comparison (0-{zoom_max:g} Hz, DC Excluded)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(0, zoom_max)
    ax.set_ylim(spec_bottom, spec_top)
    ax.legend(loc="lower right")

    # Bottom-right: theoretical Notch Filter magnitude response |H(f)|.
    ax = axes[1, 1]
    ax.plot(resp_freqs, resp_db, linewidth=1.3, color="tab:blue", label="|H(f)|")
    ax.axvline(NOTCH_FREQUENCY_HZ, color="red", linestyle="--", alpha=0.45, label=f"Notch: {NOTCH_FREQUENCY_HZ} Hz")
    ax.axhline(-3.0, color="gray", linestyle=":", linewidth=1.0, label=f"-3 dB line (BW ≈ {NOTCH_BANDWIDTH_HZ} Hz)")
    ax.set_title(f"2nd-Order Notch Filter Frequency Response (0-{zoom_max:g} Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_xlim(0, zoom_max)
    ax.set_ylim(-60, 5)
    ax.legend(loc="lower right")

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)

    fig.suptitle(
        f"Audio Filtering | Fs = {sample_rate} Hz | Duration = {duration:.2f} s",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(COMPARISON_FIGURE_PATH, dpi=150)


def main():
    # 1) Load noisy.wav generated by Student 3.
    if not NOISY_WAV_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find noisy WAV file: {NOISY_WAV_PATH.resolve()}\n"
            "Run audio_analysis.py first so noisy.wav is in the same folder."
        )

    sample_rate, raw_audio = wavfile.read(NOISY_WAV_PATH)
    noisy = to_float_audio(raw_audio)
    if noisy.ndim == 2:
        noisy = noisy.mean(axis=1)

    if len(noisy) == 0:
        raise ValueError("The WAV file contains no audio samples.")
    if not 0 < NOTCH_FREQUENCY_HZ < sample_rate / 2:
        raise ValueError(
            f"Notch frequency ({NOTCH_FREQUENCY_HZ} Hz) must be "
            f"below the Nyquist frequency ({sample_rate / 2:g} Hz)."
        )

    # Compute pole radius from desired -3 dB bandwidth: BW ≈ (1 - r) * fs / pi.
    pole_radius = 1.0 - np.pi * NOTCH_BANDWIDTH_HZ / sample_rate
    if not 0 < pole_radius < 1:
        raise ValueError(f"Pole radius r must be in (0, 1) for stability; got {pole_radius:.6f}.")

    # 2) Calculate second-order Notch Filter coefficients and apply the filter.
    w0, b, a = calculate_notch_coefficients(NOTCH_FREQUENCY_HZ, sample_rate, pole_radius)
    filtered = lfilter(b, a, noisy)

    # 3) Save filtered audio at the original sampling rate without clipping.
    save_wav(FILTERED_WAV_PATH, filtered, sample_rate)

    # 4) Compute spectra before and after filtering and measure reduction at 50 Hz.
    freqs, noisy_mag = magnitude_spectrum(noisy, sample_rate)
    _, filtered_mag = magnitude_spectrum(filtered, sample_rate)

    notch_bin = np.argmin(np.abs(freqs - NOTCH_FREQUENCY_HZ))
    measured_freq = freqs[notch_bin]
    mag_before = noisy_mag[notch_bin]
    mag_after = filtered_mag[notch_bin]
    reduction_percent = (1.0 - mag_after / mag_before) * 100.0 if mag_before > 0 else 0.0
    before_db = float(to_db(mag_before))
    after_db = float(to_db(mag_after))
    reduction_db = after_db - before_db

    # Exclude DC (0 Hz, index 0) when plotting so any DC offset does not dominate.
    plot_freqs = freqs[1:]
    noisy_db = to_db(noisy_mag[1:])
    filtered_db = to_db(filtered_mag[1:])

    # 5) Plot before/after comparison and filter frequency response.
    duration = len(noisy) / sample_rate
    plot_filtering_comparison(
        plot_freqs, noisy_db, filtered_db, b, a, sample_rate, duration,
        measured_freq, before_db, after_db, reduction_db,
    )

    # 6) Print summary.
    print(f"Sample rate (fs): {sample_rate} Hz")
    print(f"Notch frequency (f0): {NOTCH_FREQUENCY_HZ} Hz | Bandwidth (BW): {NOTCH_BANDWIDTH_HZ} Hz")
    print(f"Digital angular frequency (w0): {w0:.6f} rad/sample | Pole radius (r): {pole_radius:.6f}")
    print(f"Numerator coefficients   b = [{b[0]:.6f}, {b[1]:.6f}, {b[2]:.6f}]")
    print(f"Denominator coefficients a = [{a[0]:.6f}, {a[1]:.6f}, {a[2]:.6f}]")
    print(f"Magnitude at {measured_freq:.2f} Hz before: {mag_before:.6f} ({before_db:.2f} dB)")
    print(f"Magnitude at {measured_freq:.2f} Hz after:  {mag_after:.6f} ({after_db:.2f} dB)")
    print(f"Peak reduction at ~{NOTCH_FREQUENCY_HZ} Hz: {reduction_percent:.2f}% ({reduction_db:.2f} dB)")
    print(f"Saved: {FILTERED_WAV_PATH}, {COMPARISON_FIGURE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
