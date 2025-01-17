import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to apply FFT-based gain control to a single channel
def apply_gain_in_frequency_domain(channel, gain):
    # Convert to frequency domain using FFT
    f_shift = np.fft.fftshift(np.fft.fft2(channel))  # FFT and shift zero frequency to center
    
    # Apply gain to the magnitude (scaling the frequency components)
    magnitude = np.abs(f_shift) * gain  # Amplifying or reducing the magnitude based on gain
    phase = np.angle(f_shift)  # Retaining the phase to avoid blurring
    
    # Reconstruct the frequency domain components
    f_new = magnitude * np.exp(1j * phase)
    
    # Convert back to spatial domain using inverse FFT
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_new)))  # Inverse FFT and shift back
    
    # Normalize to valid pixel range (0-255)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    
    return img_back

# Function to equalize the histogram of a channel
def equalize_channel(channel):
    return cv2.equalizeHist(channel)

# Load and preprocess the image
def load_and_preprocess_image(image_path):
    # Load the image in BGR format
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    # Convert BGR to RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Main function to apply color balance to all images in a folder
def process_folder(input_folder, gain_factors):
    # Get all image file paths in the input folder
    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in the folder.")
        return
    
    for image_path in image_files:
        try:
            # Load the image
            image = load_and_preprocess_image(image_path)
            
            # Split the image into R, G, B channels
            r_channel, g_channel, b_channel = cv2.split(image)
            
            # Equalize each channel
            r_equalized = equalize_channel(r_channel)
            g_equalized = equalize_channel(g_channel)
            b_equalized = equalize_channel(b_channel)
            
            # Apply frequency domain color balance (gain adjustment) to each equalized channel
            r_corrected = apply_gain_in_frequency_domain(r_equalized, gain_factors['R'])
            g_corrected = apply_gain_in_frequency_domain(g_equalized, gain_factors['G'])
            b_corrected = apply_gain_in_frequency_domain(b_equalized, gain_factors['B'])
            
            # Merge corrected channels back into an image
            corrected_image = cv2.merge((r_corrected, g_corrected, b_corrected))
            
            # Display the original and corrected images
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title(f'Original: {os.path.basename(image_path)}')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(corrected_image)
            plt.title(f'Color-Balanced (Freq Domain): {os.path.basename(image_path)}')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        except FileNotFoundError as e:
            print(e)

# Path to the input folder and gain factors for each channel
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images"
gain_factors = {'R': 0.8, 'G': 1.0, 'B': 1.2}  # Adjust these values to balance the color channels

# Run the main function
process_folder(input_folder, gain_factors)
