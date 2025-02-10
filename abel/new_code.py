import cv2
import numpy as np
import os

# LAB-based white balance (global correction)
def apply_white_balance_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(lab)  # Split channels

    # Adjust the A and B channels by subtracting their mean
    a = np.clip(a - np.mean(a) + 128, 0, 255).astype(np.uint8)
    b = np.clip(b - np.mean(b) + 128, 0, 255).astype(np.uint8)

    # Merge back and convert to BGR
    lab_corrected = cv2.merge((l, a, b))
    balanced_img = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    return balanced_img

# FFT-based correction (frequency domain fine-tuning)
def apply_fft_correction(img):
    b, g, r = cv2.split(img)  # Split channels

    def process_channel(channel):
        fshift = np.fft.fft2(channel)  # Convert to frequency domain
        fshift = np.fft.fftshift(fshift)  # Shift low frequencies to center

        # Normalize by subtracting mean frequency component
        avg_val = np.mean(fshift)
        fshift = fshift - avg_val

        # Inverse FFT to get corrected image
        f_ishift = np.fft.ifftshift(fshift)
        img_corrected = np.abs(np.fft.ifft2(f_ishift))

        return img_corrected

    # Apply FFT-based correction to each channel
    b_corrected = process_channel(b)
    g_corrected = process_channel(g)
    r_corrected = process_channel(r)

    # Merge corrected channels
    corrected_img = cv2.merge((b_corrected, g_corrected, r_corrected))
    corrected_img = cv2.normalize(corrected_img, None, 0, 255, cv2.NORM_MINMAX)  # Normalize to 0-255

    return corrected_img.astype(np.uint8)

# Hybrid approach: Combining LAB and FFT corrections
def hybrid_white_balance(img):
    # Step 1: Apply LAB-based correction
    balanced_img = apply_white_balance_lab(img)

    # Step 2: Apply FFT-based fine-tuning
    final_corrected_img = apply_fft_correction(balanced_img)

    return final_corrected_img

# Load images from input folder and save corrected images to output folder
if __name__ == "__main__":
    input_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images"
    output_folder = r"C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\output_images"

    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found at {input_folder}")
        exit()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)

        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Unable to load the image {filename}. Skipping.")
            continue

        corrected_img = hybrid_white_balance(img)

        # Save the corrected image
        corrected_path = os.path.join(output_folder, filename)
        cv2.imwrite(corrected_path, corrected_img)

        # Display original and corrected images
        cv2.imshow("Original Image", img)
        cv2.imshow("Corrected Image", corrected_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
