import cv2
import numpy as np
import matplotlib.pyplot as plt

def frequency_domain_correction_and_enhancement(image, alpha=0.1, d0=30):
    """
    Applies color correction and enhancement using frequency domain techniques.
    """
    # Convert to float and split channels
    b, g, r = cv2.split(image.astype(np.float32))

    def apply_fft(channel):
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)
        magnitude = np.abs(dft_shift)
        phase = np.angle(dft_shift)
        return magnitude, phase

    # Compute FFT for each channel
    mag_r, phase_r = apply_fft(r)
    mag_g, phase_g = apply_fft(g)
    mag_b, phase_b = apply_fft(b)

    # Compute mean values for color correction
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)

    # Red & Blue channel compensation (reduce the effect of alpha to avoid blur)
    mag_r = mag_r + alpha * (mean_g - mean_r) * 0.2  # Reduce the impact further
    mag_b = mag_b + alpha * (mean_g - mean_b) * 0.2  # Same here for blue

    def apply_ifft(mag, phase):
        dft_shift = mag * np.exp(1j * phase)
        dft = np.fft.ifftshift(dft_shift)
        corrected_channel = np.fft.ifft2(dft).real
        return np.clip(corrected_channel, 0, 255).astype(np.uint8)

    # Apply inverse FFT
    r_corrected = apply_ifft(mag_r, phase_r)
    b_corrected = apply_ifft(mag_b, phase_b)

    # Merge channels to get the color corrected image
    corrected_image = cv2.merge((b_corrected, g.astype(np.uint8), r_corrected))

    # Enhance contrast using CLAHE in the LAB color space
    lab = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Reduce the CLAHE enhancement strength
    l = clahe.apply(l)
    final_image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    return final_image

# Load the input image
input_image = cv2.imread(r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images\set_f17.jpg")

# Apply frequency domain correction and enhancement
corrected_and_enhanced_image = frequency_domain_correction_and_enhancement(input_image, alpha=0.1)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(corrected_and_enhanced_image, cv2.COLOR_BGR2RGB))
plt.title("Corrected & Enhanced Image")
plt.axis('off')

plt.show()
