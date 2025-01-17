import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to apply FFT-based gain control to a single channel
def apply_gain_control(channel, gain):
    # Convert to frequency domain using FFT
    f_shift = np.fft.fftshift(np.fft.fft2(channel))
    
    # Apply gain to amplitude
    magnitude = np.abs(f_shift) * gain
    phase = np.angle(f_shift)
    f_new = magnitude * np.exp(1j * phase)
    
    # Convert back to spatial domain using inverse FFT
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_new)))
    
    # Normalize to valid pixel range (0-255)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
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

# Main function to apply gain control to all images in a folder
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
            
            # Apply gain control to each equalized channel
            r_corrected = apply_gain_control(r_equalized, gain_factors['R'])
            g_corrected = apply_gain_control(g_equalized, gain_factors['G'])
            b_corrected = apply_gain_control(b_equalized, gain_factors['B'])
            
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
            plt.title(f'Gain-Controlled and Equalized: {os.path.basename(image_path)}')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        except FileNotFoundError as e:
            print(e)

# Path to the input folder and gain factors for each channel
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images"
gain_factors = {'R': 3.0, 'G': 1.0, 'B': -0.1}

# Run the main function
process_folder(input_folder, gain_factors)
