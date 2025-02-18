import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Load the image
image = cv2.imread(r"E:\GIt\project\abel\output_images\set_f46.jpg")

# Check if the image is loaded properly
if image is None:
    print("Error: Could not load image.")
else:
    # Apply Gamma Correction
    gamma_corrected = gamma_correction(image, gamma=1.5)

    # Convert images from BGR to RGB (for correct color display)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gamma_corrected_rgb = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)

    # Display results using matplotlib
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(gamma_corrected_rgb)
    plt.title("Gamma Corrected Image")
    plt.axis("off")
    
    plt.show()
