import cv2
import numpy as np
from scipy.fftpack import dct, idct

def apply_dct(image_channel):
    block_size = 8  # DCT block size
    height, width = image_channel.shape
    dct_transformed = np.zeros_like(image_channel, dtype=np.float32)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image_channel[i:i+block_size, j:j+block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_transformed[i:i+block_size, j:j+block_size] = dct_block
    
    return dct_transformed

def apply_idct(dct_transformed):
    block_size = 8  # DCT block size
    height, width = dct_transformed.shape
    reconstructed = np.zeros_like(dct_transformed, dtype=np.float32)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = dct_transformed[i:i+block_size, j:j+block_size]
            idct_block = idct(idct(block.T, norm='ortho').T, norm='ortho')
            reconstructed[i:i+block_size, j:j+block_size] = idct_block
    
    return reconstructed

def correct_colors(image, reference_image):
    """
    Correct the colors in an image to match a reference image using DCT-based adjustments.
    """
    image = image.astype(np.float32) / 255.0
    reference_image = reference_image.astype(np.float32) / 255.0
    
    b, g, r = cv2.split(image)
    br, gr, rr = cv2.split(reference_image)

    # Apply DCT on each channel of both the input and reference image
    b_dct = apply_dct(b)
    g_dct = apply_dct(g)
    r_dct = apply_dct(r)
    
    br_dct = apply_dct(br)
    gr_dct = apply_dct(gr)
    rr_dct = apply_dct(rr)

    # Match the low-frequency components (first few DCT coefficients) between the input and reference image
    b_dct[0:4, 0:4] = br_dct[0:4, 0:4]
    g_dct[0:4, 0:4] = gr_dct[0:4, 0:4]
    r_dct[0:4, 0:4] = rr_dct[0:4, 0:4]

    # Reconstruct the image using IDCT
    b_corrected = apply_idct(b_dct)
    g_corrected = apply_idct(g_dct)
    r_corrected = apply_idct(r_dct)

    corrected_image = cv2.merge((b_corrected, g_corrected, r_corrected))
    corrected_image = np.clip(corrected_image * 255.0, 0, 255).astype(np.uint8)
    return corrected_image

def calculate_psnr(original, corrected):
    mse = np.mean((original - corrected) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

# Load images
input_image_path = r'C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images\set_f32.jpg'
hr_image_path = r'C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\hrr\set_f32hr.jpg'

input_image = cv2.imread(input_image_path)
if input_image is None:
    raise ValueError(f"Input image not found: {input_image_path}")

hr_image = cv2.imread(hr_image_path)
if hr_image is None:
    raise ValueError(f"High-resolution (HR) image not found: {hr_image_path}")

# Correct the input image based on HR image
corrected_image = correct_colors(input_image, hr_image)

# Resize all images to the same dimensions as the HR image
height, width = hr_image.shape[:2]
input_image_resized = cv2.resize(input_image, (width, height))
corrected_image_resized = cv2.resize(corrected_image, (width, height))

# Calculate PSNR values
psnr_input_hr = calculate_psnr(input_image_resized, hr_image)
psnr_corrected_hr = calculate_psnr(corrected_image_resized, hr_image)

# Combine images for side-by-side comparison
comparison = np.hstack((input_image_resized, corrected_image_resized, hr_image))

# Add labels to the comparison
font = cv2.FONT_HERSHEY_SIMPLEX
comparison = cv2.putText(comparison, "Input Image", (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
comparison = cv2.putText(comparison, "Corrected Image", (width + 50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
comparison = cv2.putText(comparison, "HR Image", (2 * width + 50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Display the results
window_title = f'Comparison | PSNR (Input-HR): {psnr_input_hr:.2f} dB | PSNR (Corrected-HR): {psnr_corrected_hr:.2f} dB'
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
cv2.imshow(window_title, comparison)

# Wait for a key press to close
cv2.waitKey(0)
cv2.destroyAllWindows()
