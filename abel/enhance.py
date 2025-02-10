import cv2
import numpy as np
import pywt

def wavelet_decomposition(image, wavelet='haar', level=1):
    """Perform wavelet decomposition."""
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    cA, (cH, cV, cD) = coeffs[0], coeffs[1]
    return cA, (cH, cV, cD)

def wavelet_reconstruction(cA, wavelet='haar', detail_coeffs=None):
    """Reconstruct image from wavelet coefficients."""
    if detail_coeffs is None:
        detail_coeffs = (None, None, None)
    coeffs = [cA, detail_coeffs]
    return pywt.waverec2(coeffs, wavelet)

def compensate_wavelet_coeffs(cA, compensation_factor=1.5):
    """Enhance approximation coefficients using compensation."""
    return cA * compensation_factor

def dehaze(image, beta=0.95):
    """Dehaze image using dark channel prior."""
    min_channel = np.min(image, axis=2)
    dark_channel = cv2.erode(min_channel, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    atmosphere = np.max(dark_channel)
    transmission = 1 - beta * (dark_channel / atmosphere)
    transmission = np.clip(transmission, 0.1, 1)
    
    restored_image = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        restored_image[:, :, c] = (image[:, :, c] - atmosphere) / transmission + atmosphere
    return np.clip(restored_image, 0, 255).astype(np.uint8)

def enhance_contrast(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced_image

def sharpen_image(image):
    """Apply unsharp mask to sharpen the image."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

def reduce_noise(image, method='bilateral', d=9, sigmaColor=75, sigmaSpace=75):
    """Reduce noise using Gaussian blur or bilateral filtering."""
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    else:
        return image

def enhance_image(image_path):
    """Enhance underwater image using WCID technique."""
    # Read input image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Wavelet Decomposition
    cA, (cH, cV, cD) = wavelet_decomposition(image)
    
    # Wavelet Coefficient Compensation
    compensated_cA = compensate_wavelet_coeffs(cA)
    
    # Reconstruct enhanced image
    enhanced_wavelet_image = wavelet_reconstruction(compensated_cA, detail_coeffs=(cH, cV, cD))
    
    # Check statistics
    print("After wavelet reconstruction:")
    print("Min:", np.min(enhanced_wavelet_image), "Max:", np.max(enhanced_wavelet_image))
    
    # Ensure the image is in the proper range and type for display
    enhanced_wavelet_image = np.uint8(np.clip(enhanced_wavelet_image, 0, 255))

    # Dehazing
    dehazed_image = dehaze(enhanced_wavelet_image)
    
    # Check statistics
    print("After dehazing:")
    print("Min:", np.min(dehazed_image), "Max:", np.max(dehazed_image))
    
    # Contrast Enhancement
    contrast_enhanced_image = enhance_contrast(dehazed_image)
    
    # Check statistics
    print("After contrast enhancement:")
    print("Min:", np.min(contrast_enhanced_image), "Max:", np.max(contrast_enhanced_image))
    
    # Sharpen the image
    sharpened_image = sharpen_image(contrast_enhanced_image)
    
    # Check statistics
    print("After sharpening:")
    print("Min:", np.min(sharpened_image), "Max:", np.max(sharpened_image))
    
    # Reduce Noise (Gaussian or Bilateral Filter)
    noise_reduced_image = reduce_noise(sharpened_image, method='bilateral')  # You can also use 'gaussian'
    
    # Check final image statistics
    print("After noise reduction:")
    print("Min:", np.min(noise_reduced_image), "Max:", np.max(noise_reduced_image))
    
    return noise_reduced_image

if __name__ == "__main__":
    # Input underwater image path
    input_image_path = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\output_images\set_f5.jpg"  # Replace with the correct path
    
    # Enhance the image
    enhanced_image = enhance_image(input_image_path)
    
    # Display the original and enhanced images
    original_image = cv2.imread(input_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Show the original image
    cv2.imshow("Original Image", original_image)
    
    # Show the enhanced image
    cv2.imshow("Enhanced Image", enhanced_image)
    
    # Wait until any key is pressed
    cv2.waitKey(0)
    
    # Close all windows
    cv2.destroyAllWindows()

    # Save the enhanced image
    output_image_path = "enhanced_underwater_image.jpg"
    cv2.imwrite(output_image_path, cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR))
    print(f"Enhanced image saved to {output_image_path}")
