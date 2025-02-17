import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the color image
image = cv2.imread(r'E:\GIt\project\abel\output_images\set_f32.jpg')

# Convert BGR to RGB for proper display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Split the image into R, G, and B channels
r, g, b = cv2.split(image_rgb)

# Apply Histogram Equalization to each channel separately
r_eq = cv2.equalizeHist(r)
g_eq = cv2.equalizeHist(g)
b_eq = cv2.equalizeHist(b)

# Merge the equalized channels back to form the contrast-enhanced color image
equalized_color = cv2.merge((r_eq, g_eq, b_eq))

# Apply Gaussian Blur to create a smoothed version
blurred = cv2.GaussianBlur(equalized_color, (5, 5), 0)

# Perform Unsharp Masking (Sharpening) using weighted sum
sharpened = cv2.addWeighted(equalized_color, 1.5, blurred, -0.5, 0)

# Display the images
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image_rgb)

# Contrast Enhanced Image
plt.subplot(1, 3, 2)
plt.title('Contrast Enhanced Image')
plt.imshow(equalized_color)

# Sharpened Image
plt.subplot(1, 3, 3)
plt.title('Sharpened Image')
plt.imshow(sharpened)

plt.show()
