import os
import cv2
import numpy as np
from skimage import color, io
from skimage.measure import shannon_entropy
from sklearn.metrics import mean_squared_error

# For BRISQUE and NIQE
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float

def calculate_entropy(image):
    """
    Calculate the Shannon entropy of an image.
    """
    return shannon_entropy(image)

def calculate_uiqm(image):
    """
    Placeholder for UIQM metric calculation.
    """
    # Replace with actual UIQM implementation
    return np.random.uniform(1, 10)  # Dummy value

def calculate_uciqe(image):
    """
    Placeholder for UCIQE metric calculation.
    """
    # Replace with actual UCIQE implementation
    return np.random.uniform(1, 10)  # Dummy value

def calculate_pcqi(image):
    """
    Placeholder for PCQI metric calculation.
    """
    # Replace with actual PCQI implementation
    return np.random.uniform(1, 10)  # Dummy value

def calculate_brisque(image):
    """
    Placeholder for BRISQUE metric calculation.
    """
    # Replace with actual BRISQUE implementation
    return np.random.uniform(1, 100)  # Dummy value

def calculate_niqe(image):
    """
    Placeholder for NIQE metric calculation.
    """
    # Replace with actual NIQE implementation
    return np.random.uniform(1, 100)  # Dummy value

def process_images(input_folder):
    results = []
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            entropy = calculate_entropy(image_gray)
            uiqm = calculate_uiqm(image)
            uciqe = calculate_uciqe(image)
            pcqi = calculate_pcqi(image)
            brisque = calculate_brisque(image)
            niqe = calculate_niqe(image)
            
            results.append({
                'Image': filename,
                'Entropy': entropy,
                'UIQM': uiqm,
                'UCIQE': uciqe,
                'PCQI': pcqi,
                'BRISQUE': brisque,
                'NIQE': niqe,
            })
    
    # Display results
    print("Image Quality Metrics:")
    print(f"{'Image':<20}{'Entropy':<10}{'UIQM':<10}{'UCIQE':<10}{'PCQI':<10}{'BRISQUE':<10}{'NIQE':<10}")
    for result in results:
        print(f"{result['Image']:<20}{result['Entropy']:<10.2f}{result['UIQM']:<10.2f}{result['UCIQE']:<10.2f}{result['PCQI']:<10.2f}{result['BRISQUE']:<10.2f}{result['NIQE']:<10.2f}")

# Set input folder
input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\output_images"  # Replace with your input folder path

process_images(input_folder)