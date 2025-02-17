import cv2
import numpy as np
import os

def contrast_stretching(img):
    """Performs contrast stretching on the RGB channels."""
    img_out = np.zeros_like(img, dtype=np.uint8)
    for i in range(3):  # Loop through R, G, B channels
        c, d = np.min(img[..., i]), np.max(img[..., i])
        a, b = 0, 255  # Desired output range
        img_out[..., i] = ((img[..., i] - c) * (b - a) / (d - c) + a).astype(np.uint8)
    return img_out

def gamma_correction(img, gamma=1.2):
    """Applies gamma correction to enhance brightness."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def rgb_to_hsi(img):
    """Converts an RGB image to HSI color space."""
    img = img.astype(np.float32) / 255
    r, g, b = cv2.split(img)
    intensity = (r + g + b) / 3
    
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = 1 - (min_rgb / (intensity + 1e-6))  # Avoid division by zero
    
    num = 0.5 * ((r - g) + (r - b))
    denom = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-6
    hue = np.arccos(num / denom)
    hue[b > g] = 2 * np.pi - hue[b > g]
    hue = hue / (2 * np.pi)  # Normalize to [0, 1]
    
    return cv2.merge((hue, saturation, intensity))

def hsi_to_rgb(hsi_img):
    """Converts an HSI image back to RGB color space."""
    hue, saturation, intensity = cv2.split(hsi_img)
    hue = hue * 2 * np.pi  # Convert back to radians
    
    r, g, b = np.zeros_like(hue), np.zeros_like(hue), np.zeros_like(hue)
    
    # Hue-based calculations
    h1 = (0 <= hue) & (hue < 2 * np.pi / 3)
    h2 = (2 * np.pi / 3 <= hue) & (hue < 4 * np.pi / 3)
    h3 = (4 * np.pi / 3 <= hue) & (hue < 2 * np.pi)
    
    r[h1] = intensity[h1] * (1 + saturation[h1] * np.cos(hue[h1]) / np.cos(np.pi / 3 - hue[h1]))
    b[h1] = intensity[h1] * (1 - saturation[h1])
    g[h1] = 3 * intensity[h1] - (r[h1] + b[h1])
    
    r[h2] = intensity[h2] * (1 - saturation[h2])
    g[h2] = intensity[h2] * (1 + saturation[h2] * np.cos(hue[h2] - 2 * np.pi / 3) / np.cos(np.pi - hue[h2]))
    b[h2] = 3 * intensity[h2] - (r[h2] + g[h2])
    
    g[h3] = intensity[h3] * (1 - saturation[h3])
    b[h3] = intensity[h3] * (1 + saturation[h3] * np.cos(hue[h3] - 4 * np.pi / 3) / np.cos(5 * np.pi / 3 - hue[h3]))
    r[h3] = 3 * intensity[h3] - (g[h3] + b[h3])
    
    rgb_img = cv2.merge((r, g, b))
    rgb_img = np.clip(rgb_img * 255, 0, 255)  # Ensure valid pixel values
    return rgb_img.astype(np.uint8)

def enhance_underwater_images(input_folder, output_folder):
    """Enhances all images in the input folder and saves them to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply contrast stretching in RGB
            stretched_img = contrast_stretching(img_rgb)
            
            # Apply gamma correction
            gamma_corrected_img = gamma_correction(stretched_img)
            
            # Convert to HSI, enhance saturation and intensity
            hsi_img = rgb_to_hsi(gamma_corrected_img)
            hsi_img[..., 1] = np.clip(hsi_img[..., 1] * 1.2, 0, 1)  # Increase saturation
            hsi_img[..., 2] = np.clip(hsi_img[..., 2] * 1.1, 0, 1)  # Increase intensity
            
            # Convert back to RGB
            enhanced_img = hsi_to_rgb(hsi_img)
            
            # Save the enhanced image
            enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, enhanced_img_bgr)
            print(f"Processed: {filename}")

# Example usage
input_folder = r"E:\GIt\project\abel\output_images"
output_folder = r"E:\GIt\project\abel\enhance_output_ver1"
enhance_underwater_images(input_folder, output_folder)
