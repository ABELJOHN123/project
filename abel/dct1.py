import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_low_pass_filter(dft_shift, cutoff_radius=15):
    """
    Applies a low-pass filter to the frequency domain representation of the image.
    """
    rows, cols = dft_shift.shape
    crow, ccol = rows // 2, cols // 2

    # Create a mask with a circular low-pass filter
    mask = np.zeros((rows, cols), np.float32)
    cv2.circle(mask, (ccol, crow), cutoff_radius, 1, thickness=-1)

    # Apply mask to the frequency domain
    dft_shift_filtered = dft_shift * mask
    return dft_shift_filtered

def apply_gain_control_in_frequency_domain(image, gain=1.1, red_gain=1.0, low_pass_radius=15):
    """
    Applies gain control in the frequency domain with noise reduction.
    """
    channels = cv2.split(image)
    corrected_channels = []

    for i, channel in enumerate(channels):
        # Perform Fourier Transform
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)

        # Apply low-pass filter to reduce high-frequency noise
        dft_shift_filtered = apply_low_pass_filter(dft_shift, cutoff_radius=low_pass_radius)

        magnitude = np.abs(dft_shift_filtered)
        phase = np.angle(dft_shift_filtered)

        # Apply different gain for the red channel
        current_gain = red_gain if i == 2 else gain
        magnitude = magnitude * current_gain

        # Reconstruct the modified DFT
        modified_dft_shift = magnitude * np.exp(1j * phase)
        modified_dft = np.fft.ifftshift(modified_dft_shift)

        # Perform Inverse Fourier Transform
        corrected_channel = np.fft.ifft2(modified_dft).real
        corrected_channel = np.clip(corrected_channel, 0, 255).astype(np.uint8)

        # Apply histogram equalization
        corrected_channel = cv2.equalizeHist(corrected_channel)

        corrected_channels.append(corrected_channel)

    corrected_image = cv2.merge(corrected_channels)
    return corrected_image

def enhance_in_frequency_domain_bgr(image, high_boost_factor=1.2, low_pass_radius=15):
    """
    Enhances the image in the frequency domain with noise reduction.
    """
    channels = cv2.split(image)
    enhanced_channels = []

    for channel in channels:
        # Perform Fourier Transform
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)

        # Apply low-pass filter to reduce high-frequency noise
        dft_shift_filtered = apply_low_pass_filter(dft_shift, cutoff_radius=low_pass_radius)

        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        radius = 50  # Radius for high-pass filter
        mask = np.ones((rows, cols), np.float32)
        cv2.circle(mask, (ccol, crow), radius, 0, thickness=-1)

        # High-pass filtering
        high_pass = dft_shift_filtered * mask
        high_boost = dft_shift_filtered + high_boost_factor * high_pass

        # Inverse Fourier Transform
        inv_dft_shift = np.fft.ifftshift(high_boost)
        enhanced_channel = np.fft.ifft2(inv_dft_shift).real
        enhanced_channel = np.clip(enhanced_channel, 0, 255).astype(np.uint8)

        enhanced_channels.append(enhanced_channel)

    enhanced_bgr = cv2.merge(enhanced_channels)
    return enhanced_bgr

def match_hr_image_to_enhanced(image, hr_image):
    """
    Matches the enhanced image to the HR image by applying histogram matching.
    """
    # Resize enhanced image if necessary
    if image.shape != hr_image.shape:
        image_resized = cv2.resize(image, (hr_image.shape[1], hr_image.shape[0]))
    else:
        image_resized = image

    matched_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2YCrCb)
    hr_image_ycrcb = cv2.cvtColor(hr_image, cv2.COLOR_BGR2YCrCb)

    # Match the Y channel (intensity)
    matched_image[:, :, 0] = cv2.equalizeHist(hr_image_ycrcb[:, :, 0])

    matched_image_bgr = cv2.cvtColor(matched_image, cv2.COLOR_YCrCb2BGR)
    return matched_image_bgr

def apply_edge_preserving_filter(image):
    """
    Apply bilateral filter for edge-preserving smoothing.
    """
    return cv2.bilateralFilter(image, 9, 75, 75)

# Set input and HR folder paths
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images"
hr_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\hrr"

input_images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
hr_images = [os.path.join(hr_folder, f) for f in os.listdir(hr_folder) if f.endswith(('.jpg', '.png'))]

# Load images
images = [cv2.imread(path) for path in input_images]
hr_images_list = [cv2.imread(path) for path in hr_images]

# Apply gain control to all images
corrected_images = [apply_gain_control_in_frequency_domain(image, gain=1.1, red_gain=1.0, low_pass_radius=15) for image in images]

# Apply enhancement to gain-controlled images
enhanced_images = [enhance_in_frequency_domain_bgr(image, high_boost_factor=1.2, low_pass_radius=15) for image in corrected_images]

# Apply edge-preserving filter after gain control to reduce blur
sharpened_images = [apply_edge_preserving_filter(image) for image in corrected_images]

# Match the enhanced images to HR images
matched_images = [match_hr_image_to_enhanced(enhanced, hr) for enhanced, hr in zip(enhanced_images, hr_images_list)]

# Display the images side by side for comparison
for i, (original, corrected, sharpened, enhanced, hr, matched) in enumerate(zip(images, corrected_images, sharpened_images, enhanced_images, hr_images_list, matched_images)):
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 6, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(f"Original {i+1}")
    plt.axis('off')

    plt.subplot(1, 6, 2)
    plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    plt.title(f"Gain-Controlled {i+1}")
    plt.axis('off')

    plt.subplot(1, 6, 3)
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.title(f"Sharpened {i+1}")
    plt.axis('off')

    plt.subplot(1, 6, 4)
    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    plt.title(f"Enhanced {i+1}")
    plt.axis('off')

    plt.subplot(1, 6, 5)
    plt.imshow(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB))
    plt.title(f"HR {i+1}")
    plt.axis('off')

    plt.subplot(1, 6, 6)
    plt.imshow(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
    plt.title(f"Matched {i+1}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
