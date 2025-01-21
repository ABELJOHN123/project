import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def compensate_red_blue_channels_frequency_domain(image, alpha=0.2):
    """
    Applies red and blue channel compensation in the frequency domain to counteract underwater light attenuation.
    """
    # Split the image into channels
    b, g, r = cv2.split(image)
    
    # Compute mean values of each channel
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)
    
    # Apply red and blue channel compensation in the frequency domain
    def frequency_domain_compensation(channel, mean_channel, mean_g, alpha):
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)
        magnitude = np.abs(dft_shift)
        phase = np.angle(dft_shift)
        
        magnitude = magnitude + alpha * (mean_g - mean_channel)
        
        modified_dft_shift = magnitude * np.exp(1j * phase)
        modified_dft = np.fft.ifftshift(modified_dft_shift)
        corrected_channel = np.fft.ifft2(modified_dft).real
        corrected_channel = np.clip(corrected_channel, 0, 255).astype(np.uint8)
        
        return corrected_channel
    
    r_corrected = frequency_domain_compensation(r, mean_r, mean_g, alpha)
    b_corrected = frequency_domain_compensation(b, mean_b, mean_g, alpha)
    
    corrected_image = cv2.merge((b_corrected, g, r_corrected))
    
    # Equalize dominant color
    lab = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    equalized_lab = cv2.merge((l, a, b))
    equalized_image = cv2.cvtColor(equalized_lab, cv2.COLOR_LAB2BGR)
    
    return equalized_image

def load_image_unicode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

def calculate_average_color(image):
    """
    Computes the overall color of an image by averaging its RGB channels.
    """
    avg_color_per_row = np.mean(image, axis=0)
    avg_color = np.mean(avg_color_per_row, axis=0)
    return avg_color.astype(int)

input_folder = r"\\?\C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images"
hr_folder = r"\\?\C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\hrr"

input_images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]
hr_images = [os.path.join(hr_folder, f) for f in os.listdir(hr_folder) if f.lower().endswith(('.jpg', '.png'))]

images = [load_image_unicode(path) for path in input_images]
hr_images_list = [load_image_unicode(path) for path in hr_images]

# Apply red and blue channel compensation and equalization in frequency domain
corrected_images = [compensate_red_blue_channels_frequency_domain(image, alpha=0.2) for image in images]

for i, (original, corrected, hr) in enumerate(zip(images, corrected_images, hr_images_list)):
    avg_color_original = calculate_average_color(original)
    avg_color_corrected = calculate_average_color(corrected)
    avg_color_hr = calculate_average_color(hr)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(f"Original {i+1}\nAvg Color: {avg_color_original}")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    plt.title(f"Color Corrected {i+1}\nAvg Color: {avg_color_corrected}")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB))
    plt.title(f"HR Image {i+1}\nAvg Color: {avg_color_hr}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
