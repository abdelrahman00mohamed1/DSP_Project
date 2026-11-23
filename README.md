# Multi-Domain Signal Processing Project (DSP)

A university Digital Signal Processing team project applying core DSP concepts across three domains: **mathematical signals**, **real audio**, and **digital images**.

Project requirements document: [`docs/DOC-20261108-WA0004.docx`](docs/DOC-20261108-WA0004.docx)

---

## Project Structure

```text
DSP_Project/
├── README.md                    # Setup, run, and contribution guide
├── requirements.txt             # Shared Python dependencies
├── .gitignore
├── docs/
│   └── DOC-20261108-WA0004.docx # Official project requirements
├── student1_dtft/               # Student 1 – Mathematical Signal & DTFT
│   └── DTFT.py
├── student2_convolution/        # Student 2 – Convolution & Difference Equations (pending)
├── student3_4_audio/            # Student 3 & 4 – Audio Analysis & Notch Filtering
│   ├── audio_analysis.py
│   ├── audio_filtering.py
│   ├── audio.wav
│   ├── noisy.wav
│   ├── filtered.wav
│   ├── audio_analysis.png
│   ├── spectrum_zoom.png
│   └── filtering_comparison.png
├── student5_2d_fourier/         # Student 5 – 2D Fourier & Image Filtering
│   ├── 2D_Fourier.py
│   └── image.png
└── student6_image_denoising/    # Student 6 – Image Noise Removal & Evaluation (pending)
```

---

## Team Progress

| Role | Topic | Folder | Status |
| :--- | :--- | :--- | :--- |
| **Student 1** | Mathematical Signal & Manual DTFT | `student1_dtft/` | Done |
| **Student 2** | Convolution & Difference Equations | `student2_convolution/` | Pending |
| **Student 3** | Audio Analysis & 50 Hz Interference | `student3_4_audio/` | Done |
| **Student 4** | Second-Order IIR Notch Filtering | `student3_4_audio/` | Done |
| **Student 5** | 2D Fourier Transform & Ideal LPF/HPF | `student5_2d_fourier/` | Done |
| **Student 6** | Image Periodic Noise Removal (MSE/PSNR) | `student6_image_denoising/` | Pending |

---

## Quick Setup (Run on Any Machine)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abdelrahman00mohamed1/DSP_Project.git
   cd DSP_Project
   ```

2. **Create a virtual environment and install dependencies:**

   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   **macOS / Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Running Each Part

Run each script **from inside its own folder** so relative input/output paths (`audio.wav`, `noisy.wav`, `image.png`) resolve automatically:

### Student 1 — DTFT
```bash
cd student1_dtft
python DTFT.py
```

### Student 3 & Student 4 — Audio Analysis & Notch Filtering
```bash
cd student3_4_audio
python audio_analysis.py
python audio_filtering.py
```

### Student 5 — 2D Fourier & Image Filtering
```bash
cd student5_2d_fourier
python 2D_Fourier.py
```

---

## How Teammates Can Add or Update Files

1. **Pull the latest changes first:**
   ```bash
   git pull origin main
   ```

2. **Place your files in your assigned folder** (for example `student2_convolution/` or `student6_image_denoising/`), keeping any input files your script reads in that same folder.

3. **If you add a new Python package**, append its name to `requirements.txt`.

4. **Commit and push your work:**
   ```bash
   git add .
   git commit -m "Add Student X: <short description>"
   git push origin main
   ```
