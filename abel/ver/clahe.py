import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the color image
image = cv2.imread(r'E:\GIt\project\abel\output_images\set_f32.jpg')

# Convert BGR to RGB for correct color display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to LAB color space
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Split LAB channels
l, a, b = cv2.split(lab)

# Apply CLAHE to the L (Lightness) channel only
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)

# Merge the processed L channel back with A and B channels
lab_clahe = cv2.merge((l_clahe, a, b))

# Convert LAB back to RGB to restore natural colors
contrast_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

# Apply Gaussian Blur for smoothing
blurred = cv2.GaussianBlur(contrast_enhanced, (5, 5), 0)

# Apply Unsharp Masking for sharpening
sharpened = cv2.addWeighted(contrast_enhanced, 1.5, blurred, -0.5, 0)

# Display the images
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image_rgb)

# Contrast Enhanced Image (CLAHE)
plt.subplot(1, 3, 2)
plt.title('Contrast Enhanced Image')
plt.imshow(contrast_enhanced)

# Sharpened Image
plt.subplot(1, 3, 3)
plt.title('Sharpened Image')
plt.imshow(sharpened)

plt.show()
