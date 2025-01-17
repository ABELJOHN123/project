import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt

def gain_control_frequency(img, gain_factor=1.5):
    # Convert the image to float32 for more precision during FFT
    img = np.float32(img)

    # Perform FFT to move to frequency domain
    f = np.fft.fftshift(np.fft.fft2(img))

    # Apply gain control (amplify frequency components)
    f_gain_controlled = f * gain_factor

    # Inverse FFT to bring the image back to the spatial domain
    img_gain_controlled = np.abs(np.fft.ifft2(np.fft.ifftshift(f_gain_controlled)))
    
    # Normalize the result to range [0, 255]
    img_gain_controlled = np.uint8(np.clip(img_gain_controlled, 0, 255))

    return img_gain_controlled

def dwt_enhancement(img, wavelet='db1', level=1):
    # Perform Discrete Wavelet Transform (DWT) for enhancement
    coeffs2 = pywt.dwt2(img, wavelet)
    LL, (LH, HL, HH) = coeffs2

    # Enhance the high-frequency details by amplifying LH, HL, and HH
    LH_enhanced = LH * 1.5  # Enhance horizontal details
    HL_enhanced = HL * 1.5  # Enhance vertical details
    HH_enhanced = HH * 1.5  # Enhance diagonal details

    # Reconstruct the image from enhanced coefficients
    img_enhanced = pywt.idwt2((LL, (LH_enhanced, HL_enhanced, HH_enhanced)), wavelet)

    # Normalize the result to range [0, 255]
    img_enhanced = np.uint8(np.clip(img_enhanced, 0, 255))

    return img_enhanced

# Load an image
img = cv2.imread(r'C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images\set_f32.jpg')  # Make sure the image path is correct

# Check if the image is loaded properly
if img is None:
    print("Error: Image not found. Please check the file path.")
else:
    # Apply Gain Control in Frequency Domain
    img_gain_controlled = gain_control_frequency(img)

    # Apply DWT for enhancement
    img_enhanced = dwt_enhancement(img_gain_controlled)

    # Display the original, gain-controlled, and enhanced images
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img_gain_controlled, cv2.COLOR_BGR2RGB))
    plt.title("Gain Controlled Image")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB))
    plt.title("DWT Enhanced Image")

    plt.show()
