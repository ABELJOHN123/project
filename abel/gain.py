import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Load and preprocess the image
def load_and_preprocess_image(image_path):
    # Load the image in BGR format
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    # Convert BGR to RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Main function to apply gain control and display the results
def main(image_path, gain_factors):
    # Load the image
    image = load_and_preprocess_image(image_path)
    
    # Split the image into R, G, B channels
    r_channel, g_channel, b_channel = cv2.split(image)
    
    # Apply gain control to each channel
    r_corrected = apply_gain_control(r_channel, gain_factors['R'])
    g_corrected = apply_gain_control(g_channel, gain_factors['G'])
    b_corrected = apply_gain_control(b_channel, gain_factors['B'])
    
    # Merge corrected channels back into an image
    corrected_image = cv2.merge((r_corrected, g_corrected, b_corrected))
    
    # Display the original and corrected images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(corrected_image)
    plt.title('Gain-Controlled Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Path to the input image and gain factors for each channel
image_path = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images\set_u648.jpg"
gain_factors = {'R': 2.0, 'G': 1.0, 'B': 1.9}

# Run the main function
try:
    main(image_path, gain_factors)
except FileNotFoundError as e:
    print(e)