import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to enhance the underwater image
def enhance_underwater_image(image):
    # Convert to float32 for better manipulation
    image_float = np.float32(image)

    # 1. Adjust contrast and brightness
    contrast = 1.3  # Increase contrast
    brightness = 20  # Increase brightness slightly
    enhanced_image = cv2.convertScaleAbs(image_float, alpha=contrast, beta=brightness)

    # 2. Denoising (remove noise using Non-Local Means Denoising)
    enhanced_image = cv2.fastNlMeansDenoisingColored(enhanced_image, None, 10, 10, 7, 21)

    # 3. Color Correction (using white balance adjustment)
    # Convert to LAB color space to adjust brightness and color balance
    lab_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)

    # Enhance L (luminance) channel for brightness
    l = cv2.equalizeHist(l)  # Apply histogram equalization to luminance channel

    # Merge the channels back
    enhanced_image = cv2.merge((l, a, b))

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

    # 4. Optional: Apply sharpness enhancement (Optional)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])  # Sharpening filter
    enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)

    # Ensure the image is within proper range
    enhanced_image = np.uint8(np.clip(enhanced_image, 0, 255))

    return enhanced_image

# Load the original underwater image
image = cv2.imread(r'C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images\set_o34.jpg')

# Enhance the image
enhanced_image = enhance_underwater_image(image)

# Display the original and enhanced image side by side
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
plt.title("Enhanced Image")
plt.axis('off')

plt.show()
