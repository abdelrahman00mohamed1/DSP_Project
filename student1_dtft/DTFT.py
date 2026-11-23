import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

HEADER = "Anas Al-Qattan Ezz El-Din | ID: 212300077"

np.random.seed(0)
fs = 1000
N = 64
n = np.arange(N)

f1, f2 = 100, 250
A1, A2 = 1.0, 0.6
noise = 0.3 * np.random.randn(N)

x = A1*np.sin(2*np.pi*f1*n/fs) + A2*np.sin(2*np.pi*f2*n/fs) + noise

K = 512
w = np.linspace(-np.pi, np.pi, K, endpoint=False)
X = np.zeros(K, dtype=complex)

for k in range(K):
    acc = 0j
    for i in range(N):
        acc += x[i] * np.exp(-1j * w[k] * i)
    X[k] = acc

mag = np.abs(X)
phase = np.angle(X)
phase_masked = np.where(mag > 0.05*mag.max(), phase, np.nan)

X_fft = np.fft.fftshift(np.fft.fft(x, K))
print("Max |difference| vs FFT:", np.max(np.abs(X - X_fft)))

pos = w >= 0
idx, _ = find_peaks(mag[pos], height=0.4*mag.max())
w_peaks = w[pos][idx]
print("Detected peaks at w (rad/sample):", np.round(w_peaks, 3))
print("Detected frequencies (Hz):", np.round(w_peaks*fs/(2*np.pi), 1))

ticks = np.pi*np.array([-1, -0.5, 0, 0.5, 1])
labels = ["-π", "-π/2", "0", "π/2", "π"]

fig, ax = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
fig.suptitle(HEADER, fontsize=11)

ax[0].stem(n, x, basefmt=" ")
ax[0].set_title("Noisy signal x[n]")
ax[0].set_xlabel("n")
ax[0].set_ylabel("Amplitude")
ax[0].grid(True)

ax[1].plot(w, mag)
ax[1].set_title("DTFT Magnitude |X(w)|")
ax[1].set_xlabel("w (rad/sample)")
ax[1].set_ylabel("|X(w)|")
ax[1].set_xticks(ticks)
ax[1].set_xticklabels(labels)
ax[1].grid(True)

ax[2].plot(w, phase_masked)
ax[2].set_title("DTFT Phase")
ax[2].set_xlabel("w (rad/sample)")
ax[2].set_ylabel("Phase (rad)")
ax[2].set_xticks(ticks)
ax[2].set_xticklabels(labels)
ax[2].grid(True)

plt.savefig("student1_dtft_plots.png", dpi=150)
plt.show()