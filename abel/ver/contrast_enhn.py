import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r'E:\GIt\project\abel\output_images\set_f5.jpg', cv2.IMREAD_COLOR)

# Convert the image to RGB (matplotlib uses RGB, while OpenCV uses BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Split the image into its three color channels
r, g, b = cv2.split(image_rgb)

# Apply Histogram Equalization on each color channel
r_equalized = cv2.equalizeHist(r)
g_equalized = cv2.equalizeHist(g)
b_equalized = cv2.equalizeHist(b)

# Merge the equalized channels back together
equalized_rgb = cv2.merge([r_equalized, g_equalized, b_equalized])

# Show the original and equalized images
plt.figure(figsize=(10, 5))

# Display the original color image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

# Display the equalized color image
plt.subplot(1, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_rgb)
plt.axis('off')

plt.show()
