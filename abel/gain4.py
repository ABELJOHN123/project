import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to balance RGB channels
def balance_colors(image):
    # Split the image into R, G, B channels
    r_channel, g_channel, b_channel = cv2.split(image)
    
    # Compute the average intensity of each channel
    r_mean, g_mean, b_mean = np.mean(r_channel), np.mean(g_channel), np.mean(b_channel)
    overall_mean = (r_mean + g_mean + b_mean) / 3.0  # Target mean for all channels

    # Normalize each channel to have the same mean intensity
    def normalize_channel(channel, mean, target_mean):
        scale_factor = target_mean / mean if mean != 0 else 1
        adjusted_channel = channel * scale_factor
        return np.clip(adjusted_channel, 0, 255).astype(np.uint8)

    r_balanced = normalize_channel(r_channel, r_mean, overall_mean)
    g_balanced = normalize_channel(g_channel, g_mean, overall_mean)
    b_balanced = normalize_channel(b_channel, b_mean, overall_mean)

    # Merge the balanced channels back into an image
    balanced_image = cv2.merge((r_balanced, g_balanced, b_balanced))
    return balanced_image

# Load and preprocess the image
def load_and_preprocess_image(image_path):
    # Load the image in BGR format
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    # Convert BGR to RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Main function to balance and display the results
def main(image_path):
    # Load the image
    image = load_and_preprocess_image(image_path)
    
    # Balance the colors
    balanced_image = balance_colors(image)
    
    # Display the original and balanced images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(balanced_image)
    plt.title('Balanced Image (No Dominant Colors)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Path to the input image
image_path = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images\set_f38.jpg"

# Run the main function
try:
    main(image_path)
except FileNotFoundError as e:
    print(e)
