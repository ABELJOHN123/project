import cv2
import numpy as np
import pywt
import os

# Function to apply DWT-based image enhancement with histogram equalization, contrast stretching, and sharpening
def dwt_image_enhancement(img):
    # Split the image into BGR channels
    b, g, r = cv2.split(img)

    def process_channel(channel):
        # Perform 2-level Discrete Wavelet Transform (DWT)
        coeffs2 = pywt.dwt2(channel, 'haar')  # Using Haar wavelet (you can change the wavelet type)
        LL, (LH, HL, HH) = coeffs2

        # Enhancement by modifying high-frequency components (amplify more aggressively)
        LH = np.multiply(LH, 2.0)  # Amplify horizontal details more aggressively
        HL = np.multiply(HL, 2.0)  # Amplify vertical details more aggressively
        HH = np.multiply(HH, 2.5)  # Amplify diagonal details more

        # Apply Histogram Equalization to improve the global contrast of the channel
        channel_equalized = cv2.equalizeHist(channel)

        # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for local contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))  # CLAHE with stronger enhancement
        channel_stretched = clahe.apply(channel_equalized)

        # Apply Unsharp Masking for clarity enhancement
        blurred = cv2.GaussianBlur(channel_stretched, (5, 5), 0)
        unsharp_mask = cv2.addWeighted(channel_stretched, 1.5, blurred, -0.5, 0)

        # Apply sharpening filter (high-pass filter) to the high-frequency components
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Simple sharpening kernel
        channel_sharpened = cv2.filter2D(unsharp_mask, -1, kernel)

        # Reconstruct the image by applying the inverse DWT with the sharpened channels
        coeffs2_enhanced = LL, (LH, HL, HH)
        channel_enhanced = pywt.idwt2(coeffs2_enhanced, 'haar')

        # Normalize to the range [0, 255]
        channel_enhanced = np.clip(channel_enhanced, 0, 255)
        channel_enhanced = channel_enhanced.astype(np.uint8)

        return channel_enhanced

    # Apply DWT enhancement, histogram equalization, contrast stretching (CLAHE), and sharpening to each color channel
    b_enhanced = process_channel(b)
    g_enhanced = process_channel(g)
    r_enhanced = process_channel(r)

    # Merge the enhanced channels back
    img_enhanced = cv2.merge((b_enhanced, g_enhanced, r_enhanced))

    return img_enhanced

# Define input and output folders
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\output_images"  # Change to your input folder
output_folder = r"C:\path\to\enhanced_image"  # Change to your output folder

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read images from the input folder, enhance them and save to the output folder
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)

    # Only process image files (skip non-image files)
    if not os.path.isfile(img_path):
        continue

    # Load image
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Unable to load the image {filename}. Skipping.")
        continue

    # Apply DWT-based image enhancement with histogram equalization, contrast, sharpening, and clarity improvement
    img_enhanced = dwt_image_enhancement(img)

    # Save the enhanced image
    enhanced_img_path = os.path.join(output_folder, filename)
    cv2.imwrite(enhanced_img_path, img_enhanced)

    # Optionally, display original and enhanced images
    cv2.imshow("Original Image", img)
    cv2.imshow("Enhanced Image (DWT + Histogram Equalization + CLAHE + Sharpening)", img_enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Image enhancement completed and saved to the output folder.")
