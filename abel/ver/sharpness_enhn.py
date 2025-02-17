import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r'E:\GIt\project\abel\output_images\set_f32.jpg')

# Apply Gaussian Blur to create the blurred image
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Perform Unsharp Masking (Sharpen the image)
sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

# Show the original and sharpened images in color
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB
plt.subplot(1, 2, 2)
plt.title('Sharpened Image')
plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB
plt.show()
