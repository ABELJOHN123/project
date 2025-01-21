import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_gain_control_in_frequency_domain(image, gain, red_gain=1.0):
    """
    Applies gain control to the image in the frequency domain and equalizes the RGB channels.

    Args:
        image (numpy.ndarray): Input BGR image.
        gain (float): Gain factor to adjust amplitude in the frequency domain for blue and green channels.
        red_gain (float): Gain factor to adjust amplitude in the frequency domain for the red channel.

    Returns:
        numpy.ndarray: Gain-corrected and equalized BGR image.
    """
    # Split the image into B, G, R channels
    channels = cv2.split(image)
    corrected_channels = []

    for i, channel in enumerate(channels):
        # Perform Fourier Transform on each channel
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)

        # Apply gain control by scaling the amplitude
        magnitude = np.abs(dft_shift)
        phase = np.angle(dft_shift)

        # Use different gain for the red channel
        current_gain = red_gain if i == 2 else gain
        magnitude = magnitude * current_gain

        # Reconstruct the frequency domain representation
        modified_dft_shift = magnitude * np.exp(1j * phase)
        modified_dft = np.fft.ifftshift(modified_dft_shift)

        # Perform Inverse Fourier Transform
        corrected_channel = np.fft.ifft2(modified_dft).real

        # Normalize the output to fit in [0, 255] range
        corrected_channel = cv2.normalize(corrected_channel, None, 0, 255, cv2.NORM_MINMAX)

        # Apply histogram equalization to improve the contrast of the channel
        corrected_channel = cv2.equalizeHist(corrected_channel.astype(np.uint8))

        corrected_channels.append(corrected_channel)

    # Merge the corrected channels back into a BGR image
    corrected_image = cv2.merge(corrected_channels)

    return corrected_image

# Set input and HR folder paths
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images"  # Replace with your input folder path
hr_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\hrr"  # Replace with your HR folder path

# Get list of images from both folders
input_images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
hr_images = [os.path.join(hr_folder, f) for f in os.listdir(hr_folder) if f.endswith(('.jpg', '.png'))]

# Print counts of images in both folders
print(f"Input images count: {len(input_images)}")
print(f"HR images count: {len(hr_images)}")

# Check if both folders have the same number of images
if len(input_images) != len(hr_images):
    print("Warning: The number of images in input_images and hr_images folders do not match.")
    # Optionally, raise an error or handle the mismatch here
    # raise ValueError("The number of images in input_images and hr_images folders must be the same.")
    # For now, continue processing the images regardless of mismatch

# Load the images
images = [cv2.imread(path) for path in input_images]
hr_images_list = [cv2.imread(path) for path in hr_images]

# Ensure images were loaded successfully
for idx, image in enumerate(images + hr_images_list):
    if image is None:
        raise FileNotFoundError(f"Image not found. Please provide a valid path.")

# Apply gain control to all images
corrected_images = [apply_gain_control_in_frequency_domain(image, gain=1.5, red_gain=0.8) for image in images]

# Display each set of images together in a single window
for i, (original, corrected, hr) in enumerate(zip(images, corrected_images, hr_images_list)):
    # Create a figure with 1 row and 3 columns (for input, corrected, and HR images)
    plt.figure(figsize=(15, 5))

    # Plot the original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(f"Original Image {i + 1}")
    plt.axis('off')

    # Plot the corrected image
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    plt.title(f"Corrected Image {i + 1}")
    plt.axis('off')

    # Plot the HR image
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB))
    plt.title(f"HR Image {i + 1}")
    plt.axis('off')

    # Display the images side by side
    plt.tight_layout()
    plt.show()
