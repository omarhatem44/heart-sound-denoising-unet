# Heart Sound Denoising using 1D U-Net

## Project Summary

This project implements a deep learning solution for denoising heart sound recordings (Phonocardiograms - PCG) using a 1D U-Net architecture.  
The model takes noisy heart audio recordings and outputs a cleaned version by suppressing background and environmental noise.  
It is intended for use in digital stethoscopes, clinical diagnostics, and medical research.

---

## Objectives

- Improve clarity in noisy PCG recordings
- Preserve diagnostic heart features such as S1, S2, and murmurs
- Enable low-latency processing for real-time applications
- Provide a reproducible, open-source medical denoising baseline

---

## Model Architecture

The model uses a 1D U-Net architecture, composed of:

- **Encoder**: Series of Conv1D + MaxPooling1D layers to extract features
- **Decoder**: UpSampling1D layers with skip connections to reconstruct the signal
- **Custom Layer**: `MatchLength1D` ensures skip connections align in time
- **Input shape**: (110250, 1) → 5 seconds at 22.05 kHz
- **Output shape**: (110250, 1)

---

## Project Structure

heart-sound-denoising-unet/
├── model/
│ └── U-Net.h5 # Saved trained model
├── data/
│ ├── noisy/ # Noisy audio files
│ └── clean/ # Clean audio files
├── train_model.py # Training script
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## Getting Started

1. Install Requirements
```bash
pip install -r requirements.txt
```
2. Run Inference
Replace `your_noisy_file.wav` with your input audio:
```bash
python src/inference.py --input your_noisy_file.wav --output denoised.wav
```
3. Train the Model (Optional)
```bash
python src/train_unet.py
```
---

## Future Improvements

-Add a real-time denoising interface
-Visualize spectrograms (before vs after)
-Add evaluation pipeline (SNR, waveform alignment)
-Create a REST API or deploy on mobile devices

---
## Author

**Omar Hatem Mohamed Emam El-Laban**  
[GitHub](https://github.com/omarhatem99) | [LinkedIn](https://www.linkedin.com/in/omar-ellaban-355ba4369)

---

## License

This project is open-source and free to use under the MIT License.



