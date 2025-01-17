import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.exposure import match_histograms

# Function to apply DFT enhancement to a single channel
def apply_dft_enhancement(channel, gain):
    # Step 1: Convert the channel to float32
    channel_float = np.float32(channel)
    
    # Step 2: Apply DFT to the channel
    dft = cv2.dft(channel_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)  # Shift zero frequency component to center
    
    # Step 3: Calculate magnitude and phase
    magnitude, angle = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
    
    # Step 4: Enhance the magnitude (apply gain)
    enhanced_magnitude = magnitude * gain
    
    # Step 5: Convert back to cartesian coordinates and apply inverse DFT
    enhanced_dft = cv2.polarToCart(enhanced_magnitude, angle)
    enhanced_dft_shift = np.fft.ifftshift(enhanced_dft)
    
    # Step 6: Perform inverse DFT to get the enhanced image
    img_back = cv2.idft(enhanced_dft_shift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Normalize the result to the range [0, 255]
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return img_back

# Function to equalize the histogram of a channel
def equalize_channel(channel):
    return cv2.equalizeHist(channel)

# Load and preprocess the image
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Calculate the average intensity of each color channel
def calculate_channel_intensity(image):
    r_channel, g_channel, b_channel = cv2.split(image)
    r_intensity = np.mean(r_channel)
    g_intensity = np.mean(g_channel)
    b_intensity = np.mean(b_channel)
    return r_intensity, g_intensity, b_intensity

# Function to determine if correction is needed
def needs_correction(r_intensity, g_intensity, b_intensity, threshold=0.1):
    overall_mean = (r_intensity + g_intensity + b_intensity) / 3
    deviations = {
        'R': abs(r_intensity - overall_mean) / overall_mean,
        'G': abs(g_intensity - overall_mean) / overall_mean,
        'B': abs(b_intensity - overall_mean) / overall_mean,
    }
    return {channel: deviation > threshold for channel, deviation in deviations.items()}

# Function to match histograms between source and reference images
def histogram_match(source, reference):
    matched = match_histograms(source, reference, channel_axis=-1)  # Updated to use channel_axis
    return matched.astype(np.uint8)

# Main function to process all images in a folder
def process_folder(input_folder, hr_folder):
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in the folder.")
        return
    
    for image_name in image_files:
        try:
            image_path = os.path.join(input_folder, image_name)
            hr_path = os.path.join(hr_folder, image_name)  # Corresponding HR image
            
            # Load the original and HR images
            image = load_and_preprocess_image(image_path)
            hr_image = load_and_preprocess_image(hr_path) if os.path.exists(hr_path) else None
            
            # Calculate channel intensities
            r_intensity, g_intensity, b_intensity = calculate_channel_intensity(image)
            
            # Determine if corrections are needed
            corrections_needed = needs_correction(r_intensity, g_intensity, b_intensity)
            
            # Split the image into R, G, B channels
            r_channel, g_channel, b_channel = cv2.split(image)
            
            # Apply corrections selectively
            if corrections_needed['R']:
                r_channel = apply_dft_enhancement(r_channel, 0.8)  # Apply DFT enhancement to Red
                r_channel = equalize_channel(r_channel)
            if corrections_needed['G']:
                g_channel = apply_dft_enhancement(g_channel, 1.2)  # Apply DFT enhancement to Green
                g_channel = equalize_channel(g_channel)
            if corrections_needed['B']:
                b_channel = apply_dft_enhancement(b_channel, 1.2)  # Apply DFT enhancement to Blue
                b_channel = equalize_channel(b_channel)
            
            # Merge adjusted channels back into an image
            corrected_image = cv2.merge((r_channel, g_channel, b_channel))
            
            # Match histograms if HR image is available
            if hr_image is not None:
                matched_image = histogram_match(corrected_image, hr_image)
            else:
                matched_image = corrected_image
            
            # Display the original, corrected, and HR images
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title(f'Original: {image_name}')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(corrected_image)
            plt.title(f'Corrected: {image_name}')
            plt.axis('off')
            
            if hr_image is not None:
                plt.subplot(1, 3, 3)
                plt.imshow(hr_image)
                plt.title(f'HR Image: {image_name}')
                plt.axis('off')
            else:
                plt.subplot(1, 3, 3)
                plt.imshow(matched_image)
                plt.title(f'Matched Image: {image_name}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        except FileNotFoundError as e:
            print(e)

# Paths to the input and HR folders
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images"
hr_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\hrr"

# Run the main function
process_folder(input_folder, hr_folder)
