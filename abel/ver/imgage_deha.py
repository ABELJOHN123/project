import cv2
import numpy as np

def dark_channel_prior(img, size=15):
    # Convert the image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Dark channel
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_channel, kernel)
    
    return dark_channel

def atmospheric_light(img, dark_channel):
    # Estimate atmospheric light
    flat_img = img.reshape((-1, 3))
    flat_dark = dark_channel.reshape((-1, 1))
    indices = np.argsort(flat_dark.flatten())[-int(0.001 * flat_dark.size):]
    brightest = flat_img[indices]
    
    A = np.mean(brightest, axis=0)
    return A

def recover_image(img, dark_channel, A, t0=0.1):
    # Recover the scene radiance
    t = 1 - t0 * (dark_channel / A[None, None])  # Broadcasting A to the shape of dark_channel
    t = np.clip(t, t0, 1)
    recovered = np.zeros_like(img, dtype=np.float32)
    
    for i in range(3):
        recovered[:, :, i] = (img[:, :, i] - A[i]) / t + A[i]
    
    return np.uint8(np.clip(recovered, 0, 255))

# Load the image
image = cv2.imread(r'E:\GIt\project\abel\output_images\set_f32.jpg')

# Apply Dark Channel Prior
dark_channel = dark_channel_prior(image)

# Estimate atmospheric light
A = atmospheric_light(image, dark_channel)

# Recover the image
recovered_image = recover_image(image, dark_channel, A)

# Show the original and dehazed image
cv2.imshow('Original Image', image)
cv2.imshow('Dehazed Image', recovered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
