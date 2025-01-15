import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in RGB format
image = cv2.imread(r'C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images\set_u649.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format

# Split the image into its Red, Green, and Blue channels
r_channel, g_channel, b_channel = cv2.split(image)

# Function to apply FFT and gain control to a single color channel
def apply_gain_control(channel, gain):
    # Step 1: Convert the image to the frequency domain using FFT
    f_transform = np.fft.fft2(channel)
    f_shift = np.fft.fftshift(f_transform)
    
    # Step 2: Apply gain to the amplitude
    magnitude = np.abs(f_shift)
    phase = np.angle(f_shift)
    magnitude *= gain
    
    # Step 3: Convert back to the spatial domain using inverse FFT
    f_new = magnitude * np.exp(1j * phase)
    f_ishift = np.fft.ifftshift(f_new)
    img_back = np.fft.ifft2(f_ishift)
    
    # Get the real part of the inverse FFT (image in the spatial domain)
    img_back = np.abs(img_back)
    
    # Normalize the result to be within valid pixel range (0-255)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    
    return img_back

# Apply gain control to each channel (adjusting the gain factors)
r_corrected = apply_gain_control(r_channel, gain=0.9)  # Increase red channel
g_corrected = apply_gain_control(g_channel, gain=1.0)  # Keep green the same
b_corrected = apply_gain_control(b_channel, gain=1.1)  # Slightly increase blue

# Merge the corrected channels back into a single image
corrected_image = cv2.merge((r_corrected, g_corrected, b_corrected))

# Display the original and corrected images using matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(corrected_image)
plt.title('Gain-Controlled Image')

plt.show()

