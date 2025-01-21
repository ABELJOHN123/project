import numpy as np
import cv2
import matplotlib.pyplot as plt

def adaptive_frequency_color_correction(image, low_gain=0.8, high_gain=1.2, threshold=30):
    """
    Perform frequency domain color correction with adaptive gain control.
    :param image: Input image (BGR)
    :param low_gain: Gain for low-frequency components
    :param high_gain: Gain for high-frequency components
    :param threshold: Frequency threshold to differentiate high and low frequencies
    :return: Color-corrected image
    """

    # Convert image to float32 for precision
    image_float = np.float32(image)

    # Split the image into R, G, B channels
    b, g, r = cv2.split(image_float)

    # Function to process each channel separately
    def process_channel(channel):
        # Apply 2D Fourier Transform
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)  # Shift zero frequency component to center

        # Compute magnitude and phase
        magnitude = np.abs(fshift)
        phase = np.angle(fshift)

        # Generate an adaptive frequency filter (high vs low frequency components)
        rows, cols = channel.shape
        crow, ccol = rows // 2 , cols // 2

        # Create a gain mask based on frequency distance from center
        x = np.arange(-ccol, ccol)
        y = np.arange(-crow, crow)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2)

        # Adaptive gain: High-frequency components boosted, low-frequency components reduced
        gain_mask = np.ones_like(magnitude)
        gain_mask[distance < threshold] = low_gain  # Reduce gain for low frequencies
        gain_mask[distance >= threshold] = high_gain  # Boost high frequencies

        # Apply the gain mask
        magnitude *= gain_mask

        # Reconstruct the frequency domain representation
        fshift_corrected = magnitude * np.exp(1j * phase)

        # Perform Inverse Fourier Transform
        f_ishift = np.fft.ifftshift(fshift_corrected)
        img_back = np.fft.ifft2(f_ishift)

        # Return the real part of the image
        return np.abs(img_back)

    # Process each RGB channel
    r_corrected = process_channel(r)
    g_corrected = process_channel(g)
    b_corrected = process_channel(b)

    # Merge corrected channels
    corrected_image = cv2.merge([b_corrected, g_corrected, r_corrected])

    # Normalize and convert back to uint8
    corrected_image_uint8 = np.uint8(np.clip(corrected_image, 0, 255))

    return corrected_image_uint8

# Load the image
image = cv2.imread(r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images\set_f46.jpg")  # Provide your image path

# Apply the adaptive color correction
corrected_image = adaptive_frequency_color_correction(image)

# Display original and corrected images
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
plt.title('Corrected Image')
plt.axis('off')

plt.show()
