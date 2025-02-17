import cv2
import numpy as np
import os

def ucm_enhancement(image):
    """Applies the Unsupervised Color Correction Method (UCM) for underwater image enhancement."""
    
    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    # 1. Enhance the Saturation Channel
    s = cv2.equalizeHist(s)  # Histogram Equalization on Saturation

    # 2. Enhance the Value (Brightness) Channel
    v = cv2.equalizeHist(v)  # Histogram Equalization on Value

    # 3. Gamma Correction on Value Channel
    gamma = 1.0  # Adjust gamma value as needed
    v = np.power(v / 255.0, gamma) * 255
    v = np.clip(v, 0, 255).astype(np.uint8)

    # 4. Merge the Enhanced Channels Back
    enhanced_hsv = cv2.merge([h, s, v])
    enhanced_rgb = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    return enhanced_rgb

# Folder paths
input_folder = r"E:\GIt\project\abel\output_images"
output_folder = r"E:\GIt\project\abel\enhance_output_ver1"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Process each image
for image_file in image_files:
    input_image_path = os.path.join(input_folder, image_file)
    output_image_path = os.path.join(output_folder, image_file)

    # Read the image
    input_image = cv2.imread(input_image_path)

    if input_image is None:
        print(f"Error: Could not read the image at {input_image_path}")
        continue  # Skip to the next image

    # Apply UCM Enhancement
    enhanced_image = ucm_enhancement(input_image)

    # Save the enhanced image
    cv2.imwrite(output_image_path, enhanced_image)
    print(f"Enhanced image saved at: {output_image_path}")

print("All images processed successfully!")
