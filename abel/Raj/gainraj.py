import cv2
import numpy as np

def gain_control(image, gain_factors):
    """
    Apply gain control to each color channel of the image.

    Parameters:
    - image: Input image (numpy array).
    - gain_factors: A tuple of gain factors for (R, G, B) channels.

    Returns:
    - Adjusted image (numpy array).
    """
    # Split the image into its color channels
    b_channel, g_channel, r_channel = cv2.split(image)

    # Apply gain control to each channel
    r_channel = cv2.multiply(r_channel, gain_factors[0])
    g_channel = cv2.multiply(g_channel, gain_factors[1])
    b_channel = cv2.multiply(b_channel, gain_factors[2])

    # Clip the values to be in the valid range [0, 255]
    r_channel = np.clip(r_channel, 0, 255).astype(np.uint8)
    g_channel = np.clip(g_channel, 0, 255).astype(np.uint8)
    b_channel = np.clip(b_channel, 0, 255).astype(np.uint8)

    # Merge the channels back together
    adjusted_image = cv2.merge((b_channel, g_channel, r_channel))

    return adjusted_image

# Load the image
image_path = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images\set_o12.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Define gain factors for R, G, B channels
gain_factors = (1.5, 1.0, 1.2)  # Example gain factors

# Apply gain control
corrected_image = gain_control(image, gain_factors)

# Display the original and corrected images
cv2.imshow('Original Image', image)
cv2.imshow('Corrected Image', corrected_image)

# Wait for a key press and close the image windows
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the corrected image
cv2.imwrite('corrected_image.jpg', corrected_image)