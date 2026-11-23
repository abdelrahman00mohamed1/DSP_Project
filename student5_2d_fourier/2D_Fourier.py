"""Student 5: 2D Fourier transform and ideal image filtering.

Requirements: OpenCV, NumPy, and Matplotlib.
Install with: pip install opencv-python numpy matplotlib
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Change this to the location of your input image.
IMAGE_PATH = Path("image.png")

# Circular-filter cutoff radius, in frequency-domain pixels.
# Try values such as 10, 30, or 60 and compare the results.
CUTOFF_RADIUS = 30


def main():
    # 1) Load the image and convert it to grayscale.
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {IMAGE_PATH.resolve()}\n"
            "Place image.png beside this script or update IMAGE_PATH."
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2) Compute and center the 2D Fourier spectrum.
    spectrum = np.fft.fft2(gray)
    spectrum_shifted = np.fft.fftshift(spectrum)
    magnitude_spectrum = np.log1p(np.abs(spectrum_shifted))

    # 3) Build ideal circular low-pass and high-pass masks.
    rows, cols = gray.shape
    center_row, center_col = rows // 2, cols // 2
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    distance = np.sqrt((u - center_col) ** 2 + (v - center_row) ** 2)
    low_pass_mask = (distance <= CUTOFF_RADIUS).astype(np.float32)
    high_pass_mask = 1.0 - low_pass_mask

    # 4) Filter the spectrum and transform each result back to the image domain.
    def apply_filter(mask):
        filtered_spectrum = spectrum_shifted * mask
        filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum))
        # The inverse should be real; discard tiny imaginary round-off values.
        return np.real(filtered_image)

    low_pass_image = apply_filter(low_pass_mask)
    high_pass_image = apply_filter(high_pass_mask)

    # 5) Display the original, spectrum, masks, and filtered images.
    titles = [
        "Original (Grayscale)",
        "Magnitude Spectrum (log)",
        f"Ideal Low-Pass Mask (D0={CUTOFF_RADIUS})",
        "Low-Pass Result",
        f"Ideal High-Pass Mask (D0={CUTOFF_RADIUS})",
        "High-Pass Result",
    ]
    images = [
        gray,
        magnitude_spectrum,
        low_pass_mask,
        low_pass_image,
        high_pass_mask,
        high_pass_image,
    ]

    plt.figure(figsize=(12, 8))
    for index, (title, displayed_image) in enumerate(zip(titles, images), start=1):
        plt.subplot(2, 3, index)
        plt.imshow(displayed_image, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
