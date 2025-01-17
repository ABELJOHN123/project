import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply frequency domain color correction
def frequency_domain_color_correction(image, red_gain=2.0, green_gain=1.8, blue_gain=2.5):
    if image is None:
        raise ValueError("Image could not be loaded. Check the file path.")

    # Split the image into BGR channels
    b, g, r = cv2.split(image)

    # Function to apply frequency domain correction to a single channel
    def apply_frequency_correction(channel, gain):
        # Perform the 2D FFT and shift the zero-frequency component to the center
        dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Get the magnitude and phase from the frequency components
        magnitude, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])

        # Apply gain to the magnitude to adjust the brightness (luminance) in the frequency domain
        magnitude *= gain  # Boosting the brightness based on the channel's gain
        magnitude = np.clip(magnitude, 0, 255)  # Clip values to avoid overflow

        # Recombine the magnitude and phase
        real, imag = cv2.polarToCart(magnitude, phase)

        # Apply inverse shift to bring the frequency components back
        dft_shift_new = cv2.merge([real, imag])

        # Perform the inverse DFT to get the corrected image in the spatial domain
        dft_ishift = np.fft.ifftshift(dft_shift_new)
        corrected_channel = cv2.idft(dft_ishift)

        # Convert the result to a format suitable for display
        corrected_channel = cv2.magnitude(corrected_channel[:, :, 0], corrected_channel[:, :, 1])

        # Normalize the result to the 0-255 range for display
        corrected_channel = np.uint8(np.clip(corrected_channel, 0, 255))

        return corrected_channel

    # Apply frequency domain color correction to each channel
    corrected_r = apply_frequency_correction(r, red_gain)
    corrected_g = apply_frequency_correction(g, green_gain)
    corrected_b = apply_frequency_correction(b, blue_gain)

    # Merge the corrected channels back into a single image
    corrected_image = cv2.merge([corrected_b, corrected_g, corrected_r])

    return corrected_image

# Read the underwater image
image = cv2.imread(r'C:\Users\albin John\OneDrive\Desktop\java\PROJECT\abel\input_images\set_f38.jpg')

# Check if the image was loaded correctly
if image is None:
    print("Error: Image could not be loaded. Check the file path.")
else:
    # Perform frequency domain color correction
    corrected_image = frequency_domain_color_correction(image)

    # Display the original and corrected images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
    plt.title('Corrected Image in Frequency Domain')
    plt.axis('off')

    plt.show()

    # Optionally save the corrected image
    cv2.imwrite('corrected_frequency_domain_image.jpg', corrected_image)
