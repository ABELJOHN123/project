import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_gain_control_in_frequency_domain(image, gain):
    """
    Applies gain control to the image in the frequency domain.

    Args:
        image (numpy.ndarray): Input BGR image.
        gain (float): Gain factor to adjust amplitude in the frequency domain.

    Returns:
        numpy.ndarray: Gain-corrected BGR image.
    """
    # Split the image into B, G, R channels
    channels = cv2.split(image)
    corrected_channels = []

    for channel in channels:
        # Perform Fourier Transform on each channel
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)

        # Apply gain control by scaling the amplitude
        magnitude = np.abs(dft_shift)
        phase = np.angle(dft_shift)
        magnitude = magnitude * gain

        # Reconstruct the frequency domain representation
        modified_dft_shift = magnitude * np.exp(1j * phase)
        modified_dft = np.fft.ifftshift(modified_dft_shift)

        # Perform Inverse Fourier Transform
        corrected_channel = np.fft.ifft2(modified_dft).real

        # Normalize the output to fit in [0, 255] range
        corrected_channel = cv2.normalize(corrected_channel, None, 0, 255, cv2.NORM_MINMAX)
        corrected_channels.append(corrected_channel.astype(np.uint8))

    # Merge the corrected channels back into a BGR image
    corrected_image = cv2.merge(corrected_channels)

    return corrected_image

# Load an example underwater image
image_path = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images\set_f5.jpg"  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not found. Please provide a valid path.")

# Apply gain control in frequency domain
gain = 1.5  # Adjust as needed
corrected_image = apply_gain_control_in_frequency_domain(image, gain)

# Display the original and corrected images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Corrected Image")
plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()
